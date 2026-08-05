# Vaticore — Project Brief

> Probabilistic energy forecasting for mini-grid and C&I solar operators in Africa.
> The name comes from the Latin *vaticinari*, to foretell, and *core*: the forecasting core that operators build their dispatch decisions on.

This document is the founding context for the build. It defines what we are making, who it is for, the technical approach, and a concrete first set of tasks. Read it top to bottom before scaffolding.

---

## 1. What we are building

Vaticore is a forecasting engine that predicts electricity **load** and **solar generation** for distributed energy operators, and turns those forecasts into operational decisions: how much battery reserve to hold, when to run a generator, and how much demand may go unserved.

We are deliberately narrow. We are not building "AI for African energy." We are building probabilistic short-term load and solar forecasting for **mini-grid operators and commercial and industrial (C&I) solar-plus-storage operators**, starting in **Nigeria and Kenya**, sold on a single measurable promise: lower fuel and battery costs through better forecasts.

Everything in this build serves that wedge. Resist scope creep back toward general-purpose energy tooling.

---

## 2. The problem

Distributed energy operators run sites that combine solar, batteries and often a diesel generator. Every day they make dispatch decisions: how much to charge the battery, when to lean on the genset, how much load to expect. These decisions are made on weak forecasts, or on rules of thumb.

The cost of a bad forecast is concrete and measurable:

- **Diesel burned** that better planning would have avoided.
- **Battery cycled** unnecessarily, shortening its life.
- **Unserved demand**, meaning customers lose power and the operator loses revenue and trust.

Operators in these markets also face a specific data reality: metering is often sparse, low-frequency, and messy, and brand-new sites have no operating history at all. Any forecasting system that assumes clean, dense, long-history data will not work here. Handling imperfect data is a first-class requirement, not an afterthought.

---

## 3. Who we serve

**Primary users:** operations and data leads at mini-grid and C&I solar operators. In practice these are the Head of Operations, Head of Data or Analytics, VP Engineering, or asset management teams.

**Target customer profile for the first pilots:** operators with enough sites to have a real data burden, ownership of their own metering data, and a commercial reason to care about accuracy. This points at the larger, better-funded players first.

**What they need from us:** a forecast they can trust and act on, expressed in terms they already use (litres of diesel, battery cycles, kWh unserved), not machine learning error metrics.

---

## 4. The product

A multi-tenant, multi-site forecasting service with three layers:

1. **Forecasting engine.** Ingests historical and live load and generation data, produces probabilistic forecasts for each site.
2. **Decision layer.** Translates the forecast into an operational recommendation: battery reserve to hold, genset advisory, expected unserved energy, each with an uncertainty range.
3. **Dashboard.** Shows forecast versus actual, the uncertainty band, and the recommendation, per site and across a fleet.

Two design commitments make this a real product rather than a dissertation rerun:

- **Probabilistic, not point forecasts.** Operators need to size reserve and decide whether to run the genset, which requires knowing the *range* of likely outcomes, not a single number. We output quantiles (for example P10, P50, P90) and evaluate on pinball loss and calibration, not just MAE.
- **Cold-start via transfer learning.** New sites have no history, which is exactly when forecasting is most valuable. The engine must produce useful forecasts for a new site from day one by transferring from similar sites, then adapt as weeks of local data arrive.

---

## 5. Technical approach

### 5.1 Forecasting methodology

The approach is grounded in prior benchmarking work (MSc dissertation, Elia Belgian grid dataset). Key findings that shape our design:

- An **ensemble of a task-adapted LLM (TIME-LLM), XGBoost and LSTM** gave the strongest accuracy.
- **Zero-shot LLM prompting (GPT-4o) failed to beat a naive persistence baseline.** We do not rely on prompting general-purpose models for forecasting. Any LLM contribution comes from task-specific adaptation, used in combination with classical learners.

Build order for the engine, simplest strong model first:

1. **Baselines.** Seasonal-naive / persistence, so every later model is measured against an honest floor.
2. **Probabilistic gradient boosting.** Quantile regression with a gradient-boosted model (for example LightGBM or XGBoost with quantile objectives) on lagged load, calendar and weather features. This is the workhorse and likely the MVP's primary model.
3. **LSTM.** Sequence model for sites with richer history.
4. **TIME-LLM.** Task-adapted foundation-model approach, added once the pipeline is proven, and central to the cold-start capability.
5. **Ensemble.** Combine the above, with weights fit on a validation set.

Do not build all five before anything works end to end. Get baseline plus quantile gradient boosting running through the full pipeline (ingest, forecast, evaluate, display) before adding the heavier models.

### 5.2 Data

- **Operator data (the goal):** each operator's own historical and live load and generation, at whatever resolution and quality they have. Design ingestion to tolerate gaps, irregular timestamps and low frequency.
- **Public and proxy data (to start, before operator data lands):** Elia dataset as a methodological baseline; Odyssey Energy Solutions and ESMAP / World Bank mini-grid datasets for realistic African load shapes; PVGIS, Open-Meteo and Solcast for weather and solar resource.
- **Weather is a required feature source.** Solar generation and load both depend on it. Wire in a weather API early.

### 5.3 Staged autonomy

We move deliberately from passive to active:

1. Day-ahead batch forecasts.
2. Rolling intraday updates (hourly).
3. Near-real-time monitoring and anomaly detection.
4. Advisory dispatch recommendations.

**We do not build closed-loop control that touches an operator's actual dispatch in the early product.** That carries liability no early-stage company should take on, and no serious operator will allow it in year one. Advisory only.

---

## 6. MVP scope

The MVP proves one thing: that Vaticore produces a probabilistic forecast good enough to beat what an operator does today, on real or realistic site data, and presents it in a way an operator can act on.

**In scope for MVP:**

- Ingest site load and generation history from CSV, with validation and a defined internal schema.
- Feature engineering: calendar features, lags, and weather from an API.
- Probabilistic forecasting: baseline plus quantile gradient boosting, producing P10 / P50 / P90 for load and solar generation, day-ahead and rolling intraday.
- Backtesting harness: pinball loss, MAE, RMSE, and calibration / coverage, against the persistence baseline.
- A decision translation: convert the forecast into a simple battery-reserve and genset advisory with an uncertainty range.
- A dashboard: forecast versus actual with uncertainty band, and the recommendation, per site.
- Multi-tenant, multi-site data model from the start.
- **Shadow mode:** the system runs alongside the operator's existing process and touches nothing.

**Explicitly out of MVP scope:** LSTM and TIME-LLM (add after the pipeline is proven), closed-loop control, billing, mobile apps, and anything that is not on the path to a shadow-mode pilot.

---

## 7. Suggested architecture

Keep the forecasting core cleanly separated from the API and the dashboard, so models can evolve without touching the interface.

```
vaticore/
  ingestion/        # CSV/API intake, validation, timestamp normalisation, gap handling
  features/         # calendar, lags, weather enrichment
  forecasting/      # models: baselines, quantile GBM, (later) LSTM, TIME-LLM, ensemble
                    # common interface: fit(), predict_quantiles(), all models interchangeable
  evaluation/       # backtesting harness, pinball loss, calibration, baseline comparison
  decisions/        # forecast -> battery reserve / genset advisory / unserved-energy estimate
  api/              # FastAPI service exposing forecasts and recommendations
  dashboard/        # forecast vs actual, uncertainty bands, recommendation
  storage/          # multi-tenant, multi-site time-series persistence
```

Design principles for the code:

- **Every model implements the same interface** (`fit`, `predict_quantiles`) so they are swappable and ensemble-able.
- **Multi-tenant and multi-site from day one.** Data is always scoped by operator and site. Retrofitting this later is painful.
- **Evaluation is a first-class module**, not a notebook. We need to prove accuracy repeatedly and defensibly, for pilots and for a research paper.

---

## 8. Suggested tech stack

- **Language:** Python for the engine (existing strength, and the ML ecosystem lives here).
- **ML:** pandas, scikit-learn, LightGBM or XGBoost (quantile), PyTorch (LSTM and TIME-LLM later). Consider a forecasting library such as Darts for baselines and backtesting scaffolding.
- **Experiment tracking:** MLflow, so model comparisons are reproducible.
- **API:** FastAPI.
- **Storage:** PostgreSQL with TimescaleDB for time series. DuckDB is fine for early local development.
- **Dashboard:** start with Streamlit for the fastest credible demo to operators, with a view to a React front end once there is a pilot. Do not sink weeks into front-end polish before the forecast is good.
- **Packaging:** Docker from early on, so a pilot deployment is not a scramble.
- **Weather data:** Open-Meteo (free) to start; Solcast or PVGIS for solar resource.

These are defaults, not dogma. Choose the simplest thing that lets the pipeline run end to end.

---

## 9. Evaluation and success metrics

Two layers, and the second matters more.

- **Model metrics:** pinball loss (primary, since we are probabilistic), plus MAE, RMSE, and calibration / coverage of the quantiles. Always reported against the persistence baseline.
- **Operator metrics (the ones that win pilots):** litres of diesel avoided, battery cycles saved, kWh of unserved energy reduced. The decision layer must be able to express results in these terms. A pilot's success criterion is agreed in operator currency, in writing, before it starts.

---

## 10. Principles and constraints

- **Narrow wedge.** Load and solar forecasting for mini-grid and C&I operators in Nigeria and Kenya. Say no to everything else for now.
- **Probabilistic by default.** Ranges, not point estimates.
- **Built for messy, sparse data.** Tolerate gaps, low frequency and cold-start. This is a differentiator, not a corner case.
- **Advisory, never closed-loop control early.** Manage liability.
- **Explainable and trustworthy.** Operators act on forecasts only if they understand and trust them. Show uncertainty honestly.
- **Evaluation-first.** Prove accuracy repeatedly and defensibly.
- **Formatting note for any generated documents and copy:** no em dashes.

---

## 11. Out of scope (for now)

Closed-loop dispatch control, billing and payments, consumer-facing apps, grid-scale utility forecasting, markets outside Nigeria and Kenya, and any model complexity added before the end-to-end pipeline works. Revisit after the first pilot produces a result.

---

## 12. Roadmap context

The build sits inside a six-month plan:

- **Months 1 to 2:** foundations and 30 operator discovery conversations. Pipeline scaffolding and baselines begin.
- **Months 3 to 4:** probabilistic engine and a demo-able dashboard. Cold-start work begins.
- **Months 5 to 6:** a shadow-mode pilot ingesting a real operator's data, and a move from daily batch to hourly rolling updates.

The near-term technical goal is a system that can take one operator's data and produce a trustworthy probabilistic forecast alongside their existing process, with results expressed in their own cost terms.

---

## 13. First build tasks

A concrete starting point. Do these in order.

1. **Scaffold the repository** using the structure in section 7, with a clean module layout and a defined internal time-series schema (operator, site, timestamp, load, generation, plus weather join keys).
2. **Build the ingestion module:** load a CSV of site load and generation, validate it, normalise timestamps, and handle missing values, into the internal schema.
3. **Wire in weather enrichment** from Open-Meteo for a site's location.
4. **Implement the baseline forecaster** (seasonal-naive / persistence) and the evaluation harness (pinball loss, MAE, RMSE, coverage) so every future model is measured against it.
5. **Implement the quantile gradient-boosting forecaster** producing P10 / P50 / P90 for load and generation, day-ahead.
6. **Build a minimal Streamlit dashboard** showing forecast versus actual with the uncertainty band for one site.
7. **Add the first decision translation:** a simple battery-reserve and genset advisory derived from the quantile forecast.

Once these run end to end on a realistic public dataset (Elia or an ESMAP mini-grid load profile), we have the MVP skeleton and can layer in the LSTM, TIME-LLM, ensemble and cold-start work.

Start with task 1.
