# CLAUDE.md — Vaticore

Working conventions for this repository. Read this before writing code. For the product vision, target users and roadmap, see `VATICORE_BRIEF.md`; this file is about *how* we build, not *what* we are building.

Vaticore is a probabilistic forecasting engine for electricity load and solar generation, for mini-grid and C&I solar operators. Forecasts must be probabilistic, the code must tolerate messy and sparse data, and evaluation is a first-class concern.

---

## Golden rules

1. **Prove the pipeline end to end before adding model complexity.** Baseline plus one quantile model running through ingest, forecast, evaluate and display beats five half-wired models. Do not build LSTM or TIME-LLM until the skeleton works.
2. **Every forecaster implements the same interface.** Models are swappable and ensemble-able. See the contract below. No model is allowed to break it.
3. **All data is scoped by `operator_id` and `site_id`.** Multi-tenant, multi-site from the first commit. Never write a function that assumes a single site.
4. **Probabilistic by default.** Models output quantiles, not point forecasts. Evaluation uses pinball loss and calibration, always against the persistence baseline.
5. **No em dashes** in any generated document, comment or copy.

---

## Tech stack

- **Python 3.11+**.
- **Dependency management:** `uv` (fast, lockfile-based). Poetry is acceptable if already set up, but prefer `uv`.
- **Lint and format:** `ruff` for both. Run before every commit.
- **Type checking:** `mypy` (or `pyright`). Type hints are required on all public functions and class methods.
- **Testing:** `pytest`, with `tests/` mirroring the package layout.
- **Data validation:** `pydantic` for config and API models; `pandera` for dataframe/time-series schema validation at ingestion boundaries.
- **ML:** `pandas`, `scikit-learn`, `lightgbm` or `xgboost` (quantile objectives), `pytorch` (LSTM and TIME-LLM, later). Consider `darts` for baseline models and backtesting scaffolding.
- **Experiment tracking:** `mlflow`. Every model comparison is logged and reproducible.
- **API:** `fastapi`.
- **Storage:** PostgreSQL with TimescaleDB for time series. `duckdb` is fine for local development and early backtests.
- **Dashboard:** `streamlit` for the first credible demo. React comes later, once there is a pilot. Do not over-invest in the front end before the forecast is good.
- **Containerisation:** Docker from early on.
- **Weather data:** Open-Meteo to start; Solcast or PVGIS for solar resource.

Defaults, not dogma. Choose the simplest option that keeps the pipeline running end to end.

---

## Repository structure

```
vaticore/
  ingestion/     # CSV/API intake, validation, timestamp normalisation, gap handling
  features/      # calendar, lags, weather enrichment
  forecasting/   # baselines, quantile GBM, (later) LSTM, TIME-LLM, ensemble
  evaluation/    # backtesting harness, pinball loss, calibration, baseline comparison
  decisions/     # forecast -> battery reserve / genset advisory / unserved energy
  api/           # FastAPI service
  dashboard/     # Streamlit app
  storage/       # multi-tenant, multi-site persistence
  config.py      # typed settings (pydantic-settings), env-driven
tests/           # mirrors the package layout
```

---

## The forecaster interface (the most important contract)

Every model in `forecasting/` subclasses a common abstract base. This is what makes models swappable and ensemble-able. Do not deviate from it.

```python
from abc import ABC, abstractmethod
import pandas as pd


class Forecaster(ABC):
    """All Vaticore forecasting models implement this interface."""

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Fit on a site's history. Must tolerate gaps and irregular timestamps."""
        ...

    @abstractmethod
    def predict_quantiles(
        self,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        """Return a frame indexed by forecast timestamp with one column per quantile."""
        ...
```

Rules:

- Output is **always quantiles**, even for the baseline (a naive model can return the same value across quantiles).
- Models must **not silently fail on missing data**. Handle gaps explicitly or raise a clear, typed error.
- The persistence baseline is the reference every other model is measured against. It always exists and always runs.

---

## Data conventions

- **Internal schema:** `operator_id`, `site_id`, `timestamp`, `load_kw`, `generation_kw`, plus weather join keys. Validate against this at every ingestion boundary with `pandera`.
- **Time is UTC internally**, always timezone-aware. Store the site's local timezone as metadata; convert only for display.
- **Resolution is explicit.** Never assume hourly. Record each site's native resolution and resample deliberately.
- **Gaps are expected.** Ingestion must detect, log and handle missing intervals. Document the gap-handling strategy in code.
- **No look-ahead leakage.** Splits are chronological. Features at time `t` use only information available at `t`. This is a hard correctness requirement, not a style preference.

---

## Evaluation conventions

- **Primary metric:** pinball (quantile) loss. **Secondary:** MAE, RMSE, and quantile calibration/coverage.
- Every evaluation run **reports the persistence baseline alongside** the model. A model that does not beat persistence is reported as such, honestly.
- The backtesting harness lives in `evaluation/` as real, tested code, not notebooks.
- Log every run to MLflow with the site, model, config and metrics, so comparisons are reproducible for both pilots and the research paper.

---

## Commands

Establish these early (as `make` targets or `uv` scripts) so every session uses the same entry points:

- `uv run pytest` — run tests.
- `uv run ruff check . && uv run ruff format .` — lint and format.
- `uv run mypy vaticore` — type check.
- `uv run streamlit run vaticore/dashboard/app.py` — launch the dashboard.
- `uv run uvicorn vaticore.api.main:app --reload` — run the API.

Run tests, lint and type check before every commit.

---

## Testing conventions

- Test the ingestion boundary hard: malformed CSVs, missing intervals, wrong timezones, duplicate timestamps. This is where real operator data will hurt, so cover it first.
- Every forecaster gets a test that it satisfies the interface and returns properly shaped quantile output.
- Test the evaluation harness on a tiny fixture with a known answer.
- Prefer small, fast, deterministic tests. Seed anything stochastic.

---

## Git and workflow conventions

- Small, focused commits with clear messages describing the change and why.
- Feature branches; keep `main` working and runnable at all times.
- Never commit secrets, API keys or operator data. `.env` is git-ignored; provide `.env.example`.
- Keep dependencies in the lockfile current and minimal.

---

## What not to do

- Do not build closed-loop control that touches an operator's real dispatch. Advisory only. This is a safety and liability boundary, not a feature to add later without discussion.
- Do not add a model that breaks the `Forecaster` interface.
- Do not write single-site code paths that ignore `operator_id` and `site_id`.
- Do not return point forecasts where a quantile forecast is expected.
- Do not let front-end polish precede a good forecast.
- Do not expand scope beyond load and solar forecasting for mini-grid and C&I operators. Note the idea, keep building the wedge.

---

## When unsure

Favour the simplest change that keeps the end-to-end pipeline running and the interface intact. If a decision affects data correctness (leakage, timezones, tenant scoping) or the safety boundary (advisory versus control), stop and flag it rather than guessing.
