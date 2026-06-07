# Results

The notebook writes its quantitative results here as it runs.

| File | Produced after | Contents |
|------|----------------|----------|
| `final_results.csv` | the core benchmark (§6–§11) | MAE, RMSE, sMAPE, and MASE for every individual model at both the 24h and 48h horizons |
| `final_results_with_tuning.csv` | tuning & ensembling (§12–§13) | the full results table extended with the Optuna/grid-search-tuned variants and the ensemble configurations |

`final_results_with_tuning.csv` is the **canonical results table** referenced
by the headline figures and by the summary in the project README.

## How results are produced

All models are evaluated on **identical, non-overlapping rolling windows**
drawn from the held-out test set, so every row in these tables is directly
comparable. Metrics are computed on the test set only — never on data seen
during training or validation.

- **MAE / RMSE** — absolute error in megawatts (lower is better)
- **sMAPE** — symmetric percentage error (scale-free)
- **MASE** — error relative to the naive-persistence baseline; **MASE < 1.0**
  means the model beats that baseline

> These CSVs are regenerated on every full run of the notebook. They are kept
> small and human-readable so the headline numbers can be inspected without
> re-executing the experiments.
