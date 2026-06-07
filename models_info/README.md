# Models & Design Decisions

This document records *what* was modelled and *why* — the rationale behind each
forecasting approach, the evaluation protocol that keeps them comparable, and
the design choices that make the study reproducible and fair. The aim of the
project is a controlled, like-for-like comparison of four modelling paradigms
on a single energy time series, not the tuning of any one model to its limit.

## Forecasting task

- **Target:** hourly total electrical load on the Belgian grid (MW)
- **Horizons:** 24 hours and 48 hours ahead
- **Input history:** up to 168 hours (one week) of recent load, depending on model

## Models by paradigm

### Statistical baselines
- **Naive Persistence** — repeats the most recent 24h profile. The reference
  point for MASE: any model worth keeping must beat it.
- **ETS (Holt-Winters)** — exponential smoothing with additive trend and daily
  seasonality.
- **ARIMA** — autoregressive integrated moving average with AIC-based order
  selection.
- **Prophet** — additive regression with daily and weekly seasonality.

### Machine learning
- **XGBoost** — gradient-boosted trees over engineered calendar features, lag
  features (t-1, t-24, t-168), and rolling 24h/168h mean and standard deviation.
  Strong, interpretable, and fast to train.

### Deep learning
- **LSTM** — a 2-layer LSTM (128 hidden units) with a 168-hour lookback,
  trained on standardised inputs.

### LLM-based
- **Time-LLM** — a **frozen GPT-2 backbone** (124M parameters) with a learnable
  reprogramming layer (~2M trainable parameters) that adapts pre-trained
  language representations for time-series via cross-attention with prompt
  prototypes ([Jin et al., 2024, ICLR](https://arxiv.org/abs/2310.01728)).
- **GPT-4o Zero-Shot** — direct numerical forecasting through natural-language
  prompts, via Azure OpenAI.
- **GPT-4o Few-Shot** — the same prompt augmented with three validation-set
  examples for in-context learning.

### Tuning & ensembling
- **Hyperparameter tuning** — Optuna for XGBoost and LSTM; grid search for
  Prophet.
- **Ensembles** — simple-average and inverse-MAE-weighted combinations of the
  top three individual models (XGBoost + LSTM + Time-LLM).

## Evaluation framework

A single evaluation harness scores every model so that results are auditable
rather than anecdotal:

- **Chronological split** — 70% train / 15% validation / 15% test, no shuffling.
- **Rolling, non-overlapping windows** across the full test set, **identical for
  every model**.
- **Four metrics** — MAE, RMSE, sMAPE, and MASE, all computed on the test set
  only.

MASE is treated as the primary metric because it is scale-free and anchored to
the naive baseline: **MASE < 1.0** means a model genuinely adds value.

## Why these choices

- **Paradigm coverage over single-model depth.** Spanning statistical, ML, deep
  learning, and LLM approaches answers the research question — *can LLMs forecast
  electricity demand?* — more honestly than over-optimising one family.
- **Frozen LLM backbone.** Time-LLM keeps GPT-2 frozen and learns only a thin
  reprogramming layer, isolating the value of *pre-trained representations* from
  brute-force retraining.
- **Identical windows and metrics for all models.** Fairness is enforced at the
  harness level, not per model, so the comparison is the deliverable.
- **Tuning reported honestly.** Tuned variants are kept even where they *fail* to
  beat their fixed-configuration counterparts — a finding in itself, not a result
  to be hidden.

## Headline outcome

A simple-average **ensemble of XGBoost + LSTM + Time-LLM** is the most accurate
configuration at both horizons. **Time-LLM** delivers the best *individual* 24h
accuracy, showing that reprogrammed language models are a viable forecasting
tool — while **direct GPT-4o prompting** is not yet competitive with
purpose-trained models. Full numbers are in
[`../results/`](../results/) and the headline table in the project README.
