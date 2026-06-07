# Source package (next phase)

This directory is reserved for the modular Python package that will refactor the
notebook's validated logic into reusable, testable components. The current
deliverable is the notebook at [`../notebook.ipynb`](../notebook.ipynb); this
package is the planned next step.

Intended modules (mirroring the notebook's pipeline stages):

- **data** — loading, hourly resampling, imputation, and the chronological
  train/validation/test split
- **features** — calendar, lag, and rolling-statistic feature engineering
- **models** — wrappers for the statistical, ML, deep-learning, and LLM
  forecasters behind a common `fit` / `predict` interface
- **evaluate** — the rolling-window harness and the MAE / RMSE / sMAPE / MASE
  metrics
- **tune** — Optuna and grid-search routines for XGBoost, LSTM, and Prophet
- **ensemble** — simple-average and inverse-MAE-weighted model combination
- **viz** — the figure-generation routines used throughout the study

Separating the package from the notebook follows standard research-engineering
practice: the notebook tells the linear story; the package makes the same logic
reusable and testable.
