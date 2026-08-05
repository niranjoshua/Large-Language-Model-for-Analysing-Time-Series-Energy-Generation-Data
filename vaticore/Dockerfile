# Production image for the Vaticore API.
# Build:  docker build -t vaticore .
# Run:    docker run -p 8000:8000 vaticore
FROM python:3.11-slim

# uv for fast, reproducible installs from the committed lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer: copy only what the install needs, so it caches across code
# changes. hatchling needs the package sources and README to build the wheel.
COPY pyproject.toml uv.lock README.md ./
COPY vaticore ./vaticore
RUN uv sync --frozen --no-dev --extra service

EXPOSE 8000

# Honour the platform provided PORT if set (Render, Railway, Fly), default 8000.
CMD ["sh", "-c", "uv run uvicorn vaticore.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
