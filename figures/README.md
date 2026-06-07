# Figures

The notebook generates and saves seventeen figures here when it runs
(`fig1_*.png` … `fig17_*.png`).

> **Note:** the same figures are **embedded directly in the executed notebook**,
> so they are visible on GitHub without rerunning anything — open
> [`../notebook.ipynb`](../notebook.ipynb). The PNGs are written to this folder
> only as a side effect of a live run.

Figures produced:

### Exploratory Data Analysis
1. `fig1_raw_timeseries.png` — full grid-load time series (2015–2025)
2. `fig2_distribution.png` — load distribution and summary statistics
3. `fig3_seasonality.png` — daily, weekly, and annual seasonal patterns
4. `fig4_missing_values.png` — missing-value locations and gap analysis
5. `fig5_decomposition.png` — seasonal-trend decomposition
6. `fig6_autocorrelation.png` — ACF / PACF correlograms

### Modelling
7. `fig7_xgboost_importance.png` — XGBoost feature importance

### Results & Comparison
8. `fig8_mae_comparison.png` — MAE across all models (grouped bar chart)
9. `fig9_mase_comparison.png` — MASE (scale-independent accuracy)
10. `fig10_heatmap.png` — all metrics, all models (heatmap)
11. `fig11_degradation.png` — horizon degradation, 24h vs 48h MAE
12. `fig12_sample_forecast.png` — sample forecast window vs actual load
13. `fig13_family_comparison.png` — accuracy by model family

### Tuning & Ensembles
14. `fig14_tuning_impact.png` — hyperparameter-tuning impact
15. `fig15_ensemble_comparison.png` — ensemble vs individual models
16. `fig16_complete_ranking.png` — complete model ranking
17. `fig17_tuning_ensemble_heatmap.png` — combined tuning & ensemble impact
