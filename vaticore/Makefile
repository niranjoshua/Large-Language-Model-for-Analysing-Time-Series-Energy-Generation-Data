.PHONY: install test lint format typecheck check api dashboard

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy vaticore

# Run before every commit.
check: lint typecheck test

api:
	uv run uvicorn vaticore.api.main:app --reload

dashboard:
	uv run streamlit run vaticore/dashboard/app.py
