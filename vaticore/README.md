# Vaticore

Probabilistic energy forecasting for mini-grid and C&I solar operators.

The name comes from the Latin *vaticinari*, to foretell, and *core*: the
forecasting core that operators build their dispatch decisions on.

Vaticore predicts electricity **load** and **solar generation** for distributed
energy operators, and turns those forecasts into operational decisions: how much
battery reserve to hold, when to run a generator, and how much demand may go
unserved. Forecasts are probabilistic, the pipeline tolerates messy and sparse
data, and evaluation is a first-class concern.

## Status

Working end to end, with a real forecasting model, an API and a dashboard.

- **Ingestion and schema**: CSV/API intake, timezone normalisation, gap
  detection, and strict validation against the internal schema.
- **Models**: a probabilistic persistence baseline and a quantile gradient
  boosting model (`quantile_gbm`), both behind one interface.
- **Evaluation**: pinball loss, calibration and a rolling origin backtest that
  always scores the candidate against persistence. On synthetic demo data the
  quantile GBM cuts pinball loss by roughly **40% on load** and **18% on solar
  generation** versus the baseline.
- **Decisions**: battery reserve and genset advisory from the quantile forecast.
- **Weather**: a real Open-Meteo client (no API key) feeding weather features.
- **Copilot**: an optional LLM layer that explains an advisory in plain language,
  grounded on the engine's numbers, with a deterministic template fallback.
- **Surfaces**: a FastAPI service, a Streamlit dashboard, and a static landing
  page, all driven by one engine facade so they never disagree.
- **Ops**: Dockerfile, docker-compose (with TimescaleDB), and a Render blueprint
  (`DEPLOY.md`).

Next: a persistent multi tenant store wired to the API, model tracking (MLflow),
and integrating weather features into the model. LSTM, TIME-LLM and an ensemble
remain stubbed against the interface.

## Layout

```
vaticore/
  ingestion/     # CSV/API intake, validation, timestamp normalisation, gap handling
  features/      # calendar, lags, weather enrichment
  forecasting/   # baselines, quantile GBM, (later) LSTM, TIME-LLM, ensemble
  evaluation/    # backtesting harness, pinball loss, calibration, baseline comparison
  decisions/     # forecast -> battery reserve / genset advisory / unserved energy
  api/           # FastAPI service (thin handlers over the engine)
  dashboard/     # Streamlit app
  copilot/       # LLM explanations grounded on the engine's numbers
  storage/       # multi-tenant, multi-site persistence
  engine.py      # orchestration facade used by api and dashboard
  datasets.py    # synthetic demo data
  config.py      # typed settings (pydantic-settings), env-driven
examples/        # runnable quickstart + real solar-data forecast
docs/landing/    # static marketing landing page (index.html)
tests/           # mirrors the package layout
Dockerfile, docker-compose.yml, render.yaml, DEPLOY.md   # deployment
```

## Quickstart

```bash
uv sync                              # core dependencies
uv run pytest                        # run the test suite
uv run python examples/quickstart.py # forecast, backtest and advise, end to end
```

Run the dashboard and the API:

```bash
uv sync --extra dashboard
uv run streamlit run vaticore/dashboard/app.py   # visual demo

uv sync --extra service
uv run uvicorn vaticore.api.main:app --reload    # API, docs at /docs
```

Example API call:

```bash
curl -X POST localhost:8000/forecast -H 'content-type: application/json' -d '{
  "operator_id": "lagos-energy", "site_id": "ikeja-minigrid",
  "target": "load_kw", "horizon": 24, "model": "quantile_gbm"
}'
```

Optional extras, installed only when needed:

```bash
uv sync --extra service     # FastAPI service
uv sync --extra dashboard   # Streamlit dashboard
uv sync --extra models      # xgboost + torch (LSTM, TIME-LLM)
uv sync --all-extras        # everything
```

Common commands are wrapped in the Makefile: `make check` runs lint, type check
and tests.

## Core contracts

Two contracts hold the system together and should not be broken:

1. **The internal schema** (`vaticore/schemas.py`). Every dataframe crossing an
   ingestion boundary is validated against it. Data is always scoped by
   `operator_id` and `site_id`, timestamps are timezone aware UTC, and targets
   are nullable so gaps are explicit rather than dropped.
2. **The forecaster interface** (`vaticore/forecasting/base.py`). Every model
   implements `fit` and `predict_quantiles`, so models are swappable and
   ensemble-able. Output is always quantiles, even for the baseline.

## Principles

- Prove the pipeline end to end before adding model complexity.
- Probabilistic by default. Evaluate on pinball loss and calibration, always
  against the persistence baseline.
- Built for messy, sparse data and cold-start sites.
- Advisory only. No closed-loop control that touches an operator's dispatch.

See `VATICORE_BRIEF.md` for the full product context and `CLAUDE.md` for build
conventions.
