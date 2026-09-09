# Locked numbers for WIP revision

## One-sentence claim

We jointly search hierarchical cascade layout \(S\) and per-occurrence IDK thresholds \(H\) under a validation accuracy floor \(\alpha\), using a memetic GA (outer) + simulated annealing (inner), and show lower expected latency \(\bar{C}\) than fixed-\(H\) layout search and thresholds-only retuning on M3N-VC **h24**.

## Acc-floor bakeoff (primary Results) — \(\alpha = 0.98337\), h24

Source: team `PAPER_4PAGE_EXPERIMENTS_AND_DRAFT.md` / `joint_bakeoff` protocol.

| Method | Joint? | Val Acc | Val cost (ms) | Holdout Acc | Holdout cost (ms) |
|--------|--------|---------|---------------|-------------|-------------------|
| A1 thresholds-only | no | 0.983376 | 1655.51 | 0.985161 | 1557.62 |
| A3 alternating | yes | 0.983376 | 1647.27 | 0.985161 | 1557.42 |
| Memetic GA (512 layouts) | yes | 0.983376 | 1627.92 | 0.985161 | 1534.71 |
| Brute force (5545) | yes | 0.983376 | 1627.23 | 0.984516 | 1530.94 |

Headline deltas:

- Joint GA vs thresholds-only: **27.59 ms** (1655.51 − 1627.92)
- GA vs exhaustive BF: **+0.69 ms** at 512/5545 layouts (~9% budget)

## In-repo EXPAND baseline (registry \(H_i\); Methods context)

Source: `checkpoints/optimizer_validation.json`, `checkpoints/baseline_comparison.json`

- EXPAND \(\mathbb{E}[C]\) (probability tables): **1194.70 ms**
- EXPAND eval (\(n=8826\)): Acc **95.63%**, mean latency **1311.70 ms**

## Classifier registry (Table I)

Source: `checkpoints/classifier_registry.json`

Report **calibrated** \(H_i\), not Timely required-precision 0.90/0.95 ([Timely, Table IV]).

Jetson Nano live cascade timing is **not** locked in this repo (`baseline_comparison.json` is CPU Windows). Experiments claim SoC only as Future Work unless Jetson logs are added.
