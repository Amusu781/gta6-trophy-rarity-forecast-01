# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Amusu781/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Amusu781/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for how many GTA VI players earn each of the 55 trophies in week one. It reads one case (a history of Exophase reads up to the checkpoint) and writes five quantiles (5/25/50/75/95%) of each trophy's earn rate at the target, 2026-11-26T00:00:00Z.

How it works: the earn rate is earners / players tracked, and Exophase only refreshes its page every few hours, so the history is a step function — flat between refreshes, jumping at each one. The model therefore forecasts the two counts separately and divides at the target. Players follow a power-law adoption curve fitted on the distinct (refreshed) reads; earners are simulated hour by hour, with each trophy's newcomer holding probability and earning hazard estimated per trophy by least squares across the observed refreshes and shrunk toward its tier mean. With a rich history that simulation median is blended (geometric mean) with a saturating curve fitted per trophy in log-time (a rising or a falling logistic, whichever the reads support). When the history is flat — no refresh seen yet — the model falls back to priors calibrated on the documented launch-night pattern: a refresh is due, the tracked base jumps several-fold, and newcomers hold almost nothing beyond the first trophy, so non-100% rates dilute hard while the 100% trophy stays put. Quantile spreads are horizon-scaled and wider for rare trophies and thin histories. Python 3.13, standard library only, deterministic, no network.

## Run it

```sh
python3 main.py
```

Reads case files from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```sh
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8a2dzbnc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
