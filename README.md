# Large Language Models for Time-Series Energy Forecasting

**Can large language models forecast electricity demand? A systematic comparison of 14 model configurations across statistical, machine learning, deep learning, and LLM-based paradigms.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![University](https://img.shields.io/badge/University%20of%20Hull-MSc%20AI%20for%20Engineering-003B5C)](https://www.hull.ac.uk/)

---

## Overview

This project investigates whether large language models (LLMs) can effectively forecast hourly electricity load on the Belgian transmission grid. Using over 395,000 quarter-hourly measurements spanning January 2015 to April 2025 from the [Elia Open Data Portal](https://opendata.elia.be/explore/dataset/ods001), the study benchmarks 14 model configurations across four paradigms and two forecast horizons (24h and 48h).

The central finding is that **LLMs can benefit energy forecasting when adapted through learned reprogramming layers** (Time-LLM achieves the best individual 24h accuracy), but **direct prompting of commercial LLMs is not yet a viable alternative** to purpose-trained models.

### Key Results

| Model | 24h MAE (MW) | 24h MASE | 48h MAE (MW) | 48h MASE |
|-------|:---:|:---:|:---:|:---:|
| **Ensemble (XGB + LSTM + Time-LLM)** | **263** | **0.49** | **299** | **0.55** |
| Time-LLM (frozen GPT-2) | 271 | 0.50 | 317 | 0.59 |
| XGBoost | 277 | 0.51 | 315 | 0.59 |
| LSTM | 302 | 0.56 | 329 | 0.61 |
| Prophet | 418 | 0.78 | 463 | 0.86 |
| Naive Persistence | 505 | 0.94 | 632 | 1.17 |
| GPT-4o (Zero-Shot) | 481 | 0.89 | 535 | 0.99 |
| GPT-4o (Few-Shot) | 775 | 1.44 | 725 | 1.35 |

> MASE < 1.0 means the model outperforms the Naive Persistence baseline.

---

## Project Structure

```
├── notebook.ipynb              # Complete experiment notebook (Google Colab)
├── README.md
├── requirements.txt
├── data/
│   └── ods001.csv              # Raw Elia dataset (download instructions below)
├── figures/
│   ├── fig1_raw_timeseries.png
│   ├── fig2_distribution.png
│   ├── fig3_seasonality.png
│   ├── fig4_missing_values.png
│   ├── fig5_decomposition.png
│   ├── fig6_autocorrelation.png
│   ├── fig7_xgboost_importance.png
│   ├── fig8_mae_comparison.png
│   ├── fig9_mase_comparison.png
│   ├── fig10_heatmap.png
│   ├── fig11_degradation.png
│   ├── fig12_sample_forecast.png
│   ├── fig13_family_comparison.png
│   ├── fig14_tuning_impact.png
│   ├── fig15_ensemble_comparison.png
│   ├── fig16_complete_ranking.png
│   └── fig17_tuning_ensemble_heatmap.png
└── results/
    └── final_results_with_tuning.csv
```

---

## Models Evaluated

### Statistical Baselines
- **Naive Persistence** — repeats the most recent 24h load profile
- **ETS (Holt-Winters)** — exponential smoothing with additive trend and daily seasonality
- **ARIMA** — autoregressive model with AIC-based order selection
- **Prophet** — Facebook's additive regression model with daily/weekly seasonality

### Machine Learning
- **XGBoost** — gradient-boosted trees with engineered calendar, lag, and rolling features

### Deep Learning
- **LSTM** — 2-layer LSTM (128 units) with 168h lookback window

### LLM-Based
- **Time-LLM** — frozen GPT-2 backbone (124M params) with learnable reprogramming layer (~2M trainable params); adapts pre-trained representations for time-series via cross-attention with prompt prototypes ([Jin et al., 2024, ICLR](https://arxiv.org/abs/2310.01728))
- **GPT-4o Zero-Shot** — direct numerical forecasting via natural language prompts
- **GPT-4o Few-Shot** — zero-shot plus 3 validation-set examples for in-context learning

### Ensemble & Tuning
- **Hyperparameter tuning** via Optuna (XGBoost, LSTM) and grid search (Prophet)
- **Simple-average and inverse-MAE-weighted ensembles** of the top 3 individual models

---

## Methodology

### Data Pipeline
1. **Raw data**: 395,000+ quarter-hourly readings (15-min resolution) from the Belgian grid
2. **Resampling**: aggregated to hourly resolution (~99,000 rows)
3. **Imputation**: linear interpolation for 736 missing values (0.19%), max gap 6h
4. **Feature engineering**: calendar features, lag features (t-1, t-24, t-168), rolling statistics (24h and 168h mean/std) for XGBoost
5. **Scaling**: StandardScaler fitted on training data only (for LSTM and Time-LLM)

### Evaluation Protocol
- **Chronological split**: 70% train / 15% validation / 15% test (no shuffling)
- **Rolling non-overlapping windows** across the full test set
- **Four metrics**: MAE, RMSE, sMAPE, MASE (all computed on the test set only)
- **Identical test windows** for every model to ensure fair comparison

### Exploratory Data Analysis
Thorough EDA covering distribution analysis, seasonal patterns (daily, weekly, annual), missing value analysis, seasonal decomposition, autocorrelation analysis (ACF/PACF), and stationarity testing (ADF), producing 6 figures that directly inform modelling decisions.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Google Colab (recommended) or local environment with GPU support

### Installation

```bash
pip install -r requirements.txt
```

### Data Download

Download the dataset from the [Elia Open Data Portal](https://opendata.elia.be/explore/dataset/ods001):
1. Navigate to the dataset page
2. Export as CSV
3. Place `ods001.csv` in the `data/` directory (or upload to Colab)

### Running the Notebook

**Google Colab (recommended):**
1. Upload `notebook.ipynb` to Google Colab
2. Upload `ods001.csv` to the Colab session
3. Run all cells sequentially

**For GPT-4o experiments**, you will need Azure OpenAI credentials stored in Colab Secrets:
- `AZURE_OPENAI_KEY`
- `AZURE_ENDPOINT`
- `AZURE_DEPLOYMENT`

> The notebook is fully self-contained. All models are trained, evaluated, and visualised within a single sequential run.

---

## Key Findings

1. **Ensembling diverse model families yields the best accuracy.** The simple-average ensemble of XGBoost + LSTM + Time-LLM outperformed every individual model at both horizons, demonstrating that model diversity matters more than individual optimisation.

2. **Reprogrammed LLMs are a viable forecasting tool.** Time-LLM shows that frozen language models contain rich, transferable pattern recognition capabilities. With only ~2M trainable parameters (vs. 124M frozen), it achieved the best individual 24h accuracy.

3. **Prompting LLMs for numerical forecasting is not yet practical.** GPT-4o's text-token interface is fundamentally mismatched to dense numerical prediction. Few-shot prompting worsened rather than improved results, likely due to tokenisation destroying numerical relationships and the absence of parameter-level learning.

4. **Automated tuning does not guarantee improvement.** Optuna-tuned XGBoost and LSTM performed worse than their fixed configurations on the test set, while Prophet improved modestly with grid search. Domain-informed defaults can be near-optimal for well-structured energy time series.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
torch>=2.0
scikit-learn>=1.3
statsmodels>=0.14
prophet>=1.1
xgboost>=2.0
transformers>=4.35
openai>=1.0
optuna>=3.4
```

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{adewumi2025llm,
  title     = {Large Language Model for Analysing Time-Series Energy Generation Data},
  author    = {Adewumi, Adeniran Ayomikun},
  year      = {2025},
  school    = {University of Hull},
  programme = {MSc Artificial Intelligence for Engineering},
  note      = {Student ID: 202447778}
}
```

---

## Acknowledgements

- **Supervisor**: Dr Zekun Guo, University of Hull
- **Data**: [Elia Transmission Belgium SA](https://opendata.elia.be/) (Elia Open Data Licence)
- **Time-LLM**: Jin et al. (2024), "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", ICLR 2024

---

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The dataset is provided under the [Elia Open Data Licence](https://opendata.elia.be/).
