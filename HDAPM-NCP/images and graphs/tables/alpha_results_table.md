# Alpha Sweep Results (with Non-Sweep Baselines)

- Label counts: positives = 3106, negatives = 4494, n = 7600
- Baselines below come from `metrics.json` (baseline and climate-only).
- Sweep rows come from `alpha_metrics.csv` (fused kernels with weight α).

| Model                         | Alpha | ROC AUC  | PR AUC  | Avg Precision |
|-------------------------------|:-----:|:--------:|:-------:|:-------------:|
| Baseline (no climate fusion)  |  —    | 0.93220  | 0.90331 | 0.90332       |
| Climate-only (no fusion)      |  —    | 0.93163  | 0.90234 | 0.90236       |
| Fused                         | 0.0   | 0.93220  | 0.90331 | 0.90332       |
| Fused                         | 0.1   | 0.93824  | 0.91457 | 0.91458       |
| Fused                         | 0.2   | 0.94263  | 0.92280 | 0.92282       |
| Fused                         | 0.3   | 0.94565  | 0.92807 | 0.92808       |
| Fused                         | 0.4   | 0.94821  | 0.93233 | 0.93234       |
| Fused                         | 0.5   | 0.95036  | 0.93591 | 0.93592       |
| Fused                         | 0.6   | 0.95207  | 0.93886 | 0.93887       |
| Fused                         | 0.7   | 0.95352  | 0.94142 | 0.94143       |
| Fused                         | 0.8   | 0.95469  | 0.94350 | 0.94351       |
| Fused                         | 0.9   | 0.95560  | 0.94517 | 0.94518       |

Notes

- “Baseline (no climate fusion)” equals α = 0.0 in the sweep by construction.
- “Climate-only (no fusion)” is the model using only the climate kernel without baseline information.
- Sweep rows are fused models: `K_fused = α*K_climate + (1−α)*K_baseline`.
