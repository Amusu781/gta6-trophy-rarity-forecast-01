#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per *.in file in the input directory (default /input, or
$MAKEFUN_INPUT_DIR) and writes one forecast per case to the output directory
(default /output, or $MAKEFUN_OUTPUT_DIR) as <id>.out:
    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

Method (written from scratch; standard library only, deterministic, no network):
  The earn rate of a trophy is earners / players_tracked.  Exophase only
  refreshes its page every few hours, so the history is a step function: flat
  between refreshes, jumping at each one.  The model therefore forecasts the
  two counts separately and divides at the target:
    * players N(t): a power-law adoption curve fitted on the distinct
      (refreshed) reads, projected to the target and capped;
    * earners E_i(t) per trophy: at each refresh, new players arrive holding
      trophy i with probability a_i, and the players already tracked earn it
      at hazard lambda_i.  Both are estimated per trophy by least squares
      across the observed refreshes, shrunk toward their tier mean, then
      simulated forward hour by hour.
  When the history is flat (no refresh seen yet) the data cannot identify
  a_i / lambda_i / N growth, so the model falls back to priors calibrated on
  the launch-night pattern the competition documents: a refresh is due
  (Exophase refreshes hours apart), the tracked base jumps, and newcomers
  hold almost nothing beyond the first trophy, so non-100% rates dilute
  hard while the 100% trophy stays put.  The spread is honest about that:
  wide when the history is thin, narrower when the history is rich.
  With a rich history the simulation median is blended (geometric mean)
  with a saturating curve fitted per trophy in log-time — a rising or a
  falling logistic, whichever the reads support.  The simulation is
  conservative (it credits newcomers only with the holding they show after
  one refresh interval, and the hazard it fits is the historical average,
  so it under-reads a cohort aging into a trophy's window); the curve alone
  can overshoot when the history is rising fast.  Their mean hedges both.
  Quantiles are the normal quantiles of the logit median with that spread.

Quantile levels: 0.05 / 0.25 / 0.5 / 0.75 / 0.95, z = -1.645 / -0.674 / 0 /
0.674 / 1.645.  Rates are clamped to [0.01, 99.99] before the logit, exactly
as the scorer does.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# --- priors and constants (tuned on the documented launch-night pattern) ---
FLAT_GAP_H = 4.5              # the documented launch-night gap: flat from
                              # 23:50Z, refreshed between 04:00Z and 04:30Z
FLAT_N_MULT_MEDIAN = 4.5      # prior median N multiplier at one refresh while
                              # the tracked base is still tiny (the one
                              # documented refresh was 6.06x); damped as the
                              # base grows, since adoption saturates
FLAT_N_LOG_SD = 0.8           # log spread of that multiplier
FLAT_HOLD_EARLY = 0.15        # newcomers hold an early (id <= 13) trophy at
FLAT_HOLD_LATE = 0.02         # ... this fraction of its current rate; late ones at
FLAT_LAMBDA_PER_DAY = 0.01    # prior earning hazard for tracked players
SHRINK_M = 2.0                # pseudo-count for shrinkage toward tier means
RW_SIGMA_PER_SQRT_DAY = 0.5   # random-walk spread, as the house baseline uses
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.60
N_GROWTH_CAP = 8.0            # never project N beyond 8x its last value
MAX_HOURS_STEP = 1.0          # simulation step, hours


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lo):
    if lo > 20.0:
        return 100.0
    if lo < -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lo))


def sat_curve_median(states, t0, target, i):
    """A saturating curve in log-time, fitted to one trophy's distinct reads.

    Rates move like a cohort-aging curve: they rise (players reach the
    trophy) or fall (newcomers dilute them) and then level off.  Fit
    r(t) = r_inf * sigmoid(direction * (log1p(hours) - m) / s) with the
    direction (rising or falling) chosen by least squares, r_inf by linear
    least squares given the shape, and (m, s) by grid search; read the
    curve at the target.  Returns None when the history is too thin to fit.
    """
    xs, ys = [], []
    for (st, _n, srates) in states:
        r = srates[i]
        if r is None:
            continue
        h = max(0.0, (st - t0).total_seconds() / 3600.0)
        xs.append(math.log1p(h))
        ys.append(r)
    if len(xs) < 4:
        return None
    best = None
    for direction in (1, -1):
        for mi in range(0, 33):
            m = mi * 0.25
            for sg in (0.3, 0.5, 0.8, 1.2, 2.0):
                shape = []
                for x in xs:
                    arg = direction * (x - m) / sg
                    z = 1.0 / (1.0 + math.exp(-arg)) if abs(arg) < 30 else (1.0 if arg > 0 else 0.0)
                    shape.append(z)
                num = sum(z * y for z, y in zip(shape, ys))
                den = sum(z * z for z in shape)
                if den <= 0:
                    continue
                r_inf = min(99.99, max(0.01, num / den))
                rss = sum((y - r_inf * z) ** 2 for z, y in zip(shape, ys))
                if best is None or rss < best[0]:
                    best = (rss, r_inf, m, sg, direction)
    if best is None:
        return None
    _, r_inf, m, sg, direction = best
    h_t = max(0.0, (target - t0).total_seconds() / 3600.0)
    arg = direction * (math.log1p(h_t) - m) / sg
    z = 1.0 / (1.0 + math.exp(-arg)) if abs(arg) < 30 else (1.0 if arg > 0 else 0.0)
    return min(99.99, max(0.01, r_inf * z))


def quantiles_from_logit(median_logit, sigma):
    out = []
    for z in Z:
        q = inv_logit_pct(median_logit + z * sigma)
        out.append(round(min(100.0, max(0.0, q)), 4))
    # enforce non-decreasing after rounding
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def least_squares(xs, ys):
    """Slope and intercept of y ~ a + b*x; returns (b, a)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def distinct_states(history):
    """Collapse consecutive identical reads to one state each.

    Returns a list of (time, players, rates) with at least one entry.
    """
    states = []
    for h in history:
        t = parse_time(h["at"])
        n = h["players_tracked"]
        rates = h["rates"]
        if states and states[-1][1] == n and states[-1][2] == rates:
            continue
        states.append((t, n, rates))
    return states


def forecast_players(states, target, horizon_days):
    """Project players_tracked at the target.

    Power-law adoption curve on the distinct reads when there are enough of
    them; a prior multiplier when the history is flat.
    Returns (n_target, log_sd) where log_sd is the log-scale uncertainty.
    """
    n_last = states[-1][1]
    t0 = states[0][0]
    pts = [( (t - t0).total_seconds() / 3600.0, n) for (t, n, _) in states]
    distinct_n = sorted(set(n for _, n in pts))
    if len(distinct_n) >= 3:
        xs = [math.log1p(h) for h, _ in pts]
        ys = [math.log(n) for _, n in pts]
        b, a = least_squares(xs, ys)
        b = max(0.0, b)  # adoption never shrinks in the fit
        h_t = (target - t0).total_seconds() / 3600.0
        n_t = math.exp(a + b * math.log1p(h_t))
        # residual spread of the fit, as log uncertainty
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 2))
        return min(n_t, N_GROWTH_CAP * n_last), max(0.15, sd)
    if len(distinct_n) == 2:
        (h1, n1), (h2, n2) = pts[0], pts[-1]
        if h2 > h1 and n2 > n1:
            rate = math.log(n2 / n1) / (h2 - h1)  # per hour, log scale
            h_t = (target - t0).total_seconds() / 3600.0
            n_t = n2 * math.exp(rate * (h_t - h2) * 0.7)  # damped
            return min(n_t, N_GROWTH_CAP * n_last), 0.35
        return float(n_last), 0.35
    # flat history: no refresh seen. The page has been flat since the first
    # read, so the last refresh was at or before it; the next one is due
    # after about FLAT_GAP_H, and while the tracked base is tiny each
    # refresh multiplies it several-fold (the documented one: 6.06x).
    first_t = states[0][0]
    last_t = states[-1][0]
    elapsed_flat_h = (last_t - first_t).total_seconds() / 3600.0
    time_to_refresh = max(0.0, FLAT_GAP_H - elapsed_flat_h)
    horizon_h = horizon_days * 24.0
    if time_to_refresh <= horizon_h:
        expected_refreshes = 1.0 + max(0.0, horizon_h - time_to_refresh) / FLAT_GAP_H
    else:
        expected_refreshes = horizon_h / FLAT_GAP_H
    mult_median = FLAT_N_MULT_MEDIAN * (638.0 / max(n_last, 1)) ** 0.3
    mult_median = max(1.2, mult_median)
    mult = mult_median ** expected_refreshes
    mult = min(mult, N_GROWTH_CAP)
    return n_last * mult, FLAT_N_LOG_SD * math.sqrt(max(expected_refreshes, 0.25))


def estimate_dynamics(states, n_trophies, tiers):
    """Estimate per-trophy (a_i, lambda_i) from the observed refreshes.

    a_i: share of newly tracked players holding trophy i.
    lambda_i: per-day hazard that a tracked player without it earns it.
    Both shrunk toward their tier mean. Returns (a, lam, n_refreshes).
    """
    n_states = len(states)
    a_raw = [None] * n_trophies
    lam_raw = [None] * n_trophies
    n_ref = 0
    if n_states >= 2:
        steps = []
        for k in range(1, n_states):
            t_prev, n_prev, r_prev = states[k - 1]
            t_cur, n_cur, r_cur = states[k]
            dn = n_cur - n_prev
            dt_days = (t_cur - t_prev).total_seconds() / 86400.0
            if dn == 0 and dt_days <= 0:
                continue
            steps.append((n_prev, r_prev, n_cur, r_cur, dn, dt_days))
        n_ref = len(steps)
        if n_ref >= 1:
            for i in range(n_trophies):
                num = 0.0
                den = 0.0
                for (n_prev, r_prev, n_cur, r_cur, dn, dt) in steps:
                    if r_prev[i] is None or r_cur[i] is None:
                        continue
                    de = r_cur[i] * n_cur / 100.0 - r_prev[i] * n_prev / 100.0
                    num += de * dn
                    den += dn * dn
                if den > 0:
                    a_raw[i] = min(1.0, max(0.0, num / den))
            # lambda from the residual after newcomer holdings
            for i in range(n_trophies):
                if a_raw[i] is None:
                    continue
                num = 0.0
                den = 0.0
                for (n_prev, r_prev, n_cur, r_cur, dn, dt) in steps:
                    if r_prev[i] is None or r_cur[i] is None or dt <= 0:
                        continue
                    e_prev = r_prev[i] * n_prev / 100.0
                    e_cur = r_cur[i] * n_cur / 100.0
                    resid = (e_cur - e_prev) - a_raw[i] * dn
                    pool = max(0.0, n_prev - e_prev)
                    num += resid
                    den += pool * dt
                if den > 0:
                    lam_raw[i] = min(0.3, max(0.0, num / den))
    # tier means for shrinkage
    def tier_mean(vals):
        out = {}
        for tier in set(tiers):
            xs = [v for v, t in zip(vals, tiers) if v is not None and t == tier]
            out[tier] = sum(xs) / len(xs) if xs else None
        glob = [v for v in vals if v is not None]
        g = sum(glob) / len(glob) if glob else None
        return out, g

    a_tier, a_glob = tier_mean(a_raw)
    lam_tier, lam_glob = tier_mean(lam_raw)
    a = []
    lam = []
    for i in range(n_trophies):
        tier = tiers[i]
        # newcomer holding probability (tier mean when this trophy's own
        # refreshes do not identify it)
        if a_raw[i] is not None:
            prior = a_tier.get(tier) if a_tier.get(tier) is not None else a_glob
            if prior is None:
                prior = 0.05
            a.append((n_ref * a_raw[i] + SHRINK_M * prior) / (n_ref + SHRINK_M))
        else:
            prior = a_tier.get(tier) if a_tier.get(tier) is not None else a_glob
            a.append(0.05 if prior is None else prior)
        if lam_raw[i] is not None:
            prior = lam_tier.get(tier) if lam_tier.get(tier) is not None else lam_glob
            if prior is None:
                prior = FLAT_LAMBDA_PER_DAY
            lam.append((n_ref * lam_raw[i] + SHRINK_M * prior) / (n_ref + SHRINK_M))
        else:
            prior = lam_tier.get(tier) if lam_tier.get(tier) is not None else lam_glob
            lam.append(FLAT_LAMBDA_PER_DAY if prior is None else prior)
    return a, lam, n_ref


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    tiers = [t["tier"] for t in trophies]
    history = case["history"]
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = (target - at).total_seconds() / 86400.0
    horizon_hours = horizon_days * 24.0

    states = distinct_states(history)
    t_last, n_last, r_last = states[-1]
    n_target, n_log_sd = forecast_players(states, target, horizon_days)
    a_list, lam_list, n_ref = estimate_dynamics(states, n_tr, tiers)

    # Flat history: the tracked base at a refresh (median multiplier,
    # damped as the base grows, since adoption saturates).
    if n_ref == 0:
        mult_median = FLAT_N_MULT_MEDIAN * (638.0 / max(n_last, 1)) ** 0.3
        mult_median = max(1.2, mult_median)
        n_at_refresh = n_last * mult_median
    else:
        n_at_refresh = None

    # last earners per trophy
    e_last = []
    for i in range(n_tr):
        r = r_last[i]
        e_last.append(None if r is None else r * n_last / 100.0)

    rows = []
    for i in range(n_tr):
        r = r_last[i]
        if r is None:
            # trophy missing from the last read: fall back to the last read
            # that has it, else a small rate
            r = 0.5
            for (_, _, rs) in reversed(states):
                if rs[i] is not None:
                    r = rs[i]
                    break
        # the 100% trophy: every tracked player holds it; it stays put
        if r >= 99.99:
            rows.append({"id": trophies[i]["id"],
                         "q": [99.5, 99.9, 100.0, 100.0, 100.0]})
            continue

        # --- median via the earners/players simulation ---
        if n_ref >= 1 and a_list[i] is not None and lam_list[i] is not None:
            a_i = a_list[i]
            lam_i = lam_list[i]
            # simulate N along the fitted path: interpolate log N from the
            # power-law fit by reusing the forecast ratio hour by hour
            e = e_last[i] if e_last[i] is not None else r * n_last / 100.0
            n_cur = float(n_last)
            steps = max(1, int(math.ceil(horizon_hours / MAX_HOURS_STEP)))
            dt_h = horizon_hours / steps
            dt_d = dt_h / 24.0
            # per-step N multiplier toward n_target (geometric)
            g_step = (n_target / n_cur) ** (1.0 / steps) if n_target > n_cur else 1.0
            for _ in range(steps):
                n_next = n_cur * g_step
                dn = n_next - n_cur
                e = e + a_i * dn + lam_i * max(0.0, n_cur - e) * dt_d
                e = min(e, n_next)
                n_cur = n_next
            median = 100.0 * e / n_cur if n_cur > 0 else r
        else:
            # flat history: no refresh observed. A refresh is due (the page
            # has been flat for hours and refreshes come hours apart), so the
            # median assumes one: the tracked base jumps several-fold and
            # newcomers hold almost nothing beyond the first trophy, so
            # rates dilute. The spread stays wide, because the refresh may
            # also land after the target and leave the rates frozen.
            hold = FLAT_HOLD_EARLY if trophies[i]["id"] <= 13 else FLAT_HOLD_LATE
            e_new = hold * r / 100.0 * (n_at_refresh - n_last)
            e_earn = FLAT_LAMBDA_PER_DAY * max(0.0, n_last - r * n_last / 100.0) * horizon_days
            e_t = r * n_last / 100.0 + e_new + e_earn
            median = 100.0 * e_t / n_at_refresh if n_at_refresh > 0 else r
            # wide spread: the refresh timing and size are both uncertain
            sigma = math.sqrt(RW_SIGMA_PER_SQRT_DAY ** 2 * max(horizon_days, 0.0)
                              + FLAT_N_LOG_SD ** 2)
            sigma = max(sigma, 0.90)
            sigma = min(sigma, SIGMA_CAP)
            rows.append({"id": trophies[i]["id"],
                         "q": quantiles_from_logit(logit_pct(min(99.99, max(0.01, median))), sigma)})
            continue

        median = min(99.99, max(0.01, median))
        # Blend with the saturating log-time curve: the simulation is
        # conservative (it credits newcomers only with the holding they
        # show after one refresh interval, and the hazard it fits is the
        # historical average, so it under-reads a cohort aging into a
        # trophy's window), while the curve alone can overshoot when the
        # history is rising fast.  Their geometric mean hedges both ways.
        sat = sat_curve_median(states, states[0][0], target, i)
        if sat is not None:
            median = math.sqrt(median * sat)
            median = min(99.99, max(0.01, median))

        # --- spread (rich history only; the flat case used its own wide band) ---
        # random-walk term grows with the horizon, as the house baseline has it
        sigma_rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
        # count noise: rare trophies (few earners) are noisier on the logit scale
        e_now = e_last[i] if e_last[i] is not None else r * n_last / 100.0
        sigma_count = min(0.5, 1.0 / math.sqrt(max(e_now, 1.0)))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_count ** 2)
        sigma = max(sigma, SIGMA_FLOOR)
        sigma = min(sigma, SIGMA_CAP)

        rows.append({"id": trophies[i]["id"],
                     "q": quantiles_from_logit(logit_pct(median), sigma)})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    if len(argv) > 1:
        in_dir = argv[1]
    if len(argv) > 2:
        out_dir = argv[2]
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        sys.stderr.write("trophy forecast: no .in case file in %s\n" % in_dir)
        return 2
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        sys.stderr.write("trophy forecast: %s -> %s (%d trophies, %d bytes)\n"
                         % (name, out_path, len(forecast["trophies"]), len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
