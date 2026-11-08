# Build context is the repository root (see infra/compose.yaml).
FROM python:3.13-slim AS base

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf

RUN pip install --no-cache-dir uv==0.13.7 \
    && useradd --create-home --uid 1000 mezvuat \
    && mkdir -p /cache/hf /data && chown -R mezvuat:mezvuat /cache /data

WORKDIR /app/backend

COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN uv sync --frozen --no-default-groups --group ml --no-install-project

COPY backend/src ./src
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./
RUN uv sync --frozen --no-default-groups --group ml --no-editable

COPY data/manifests /app/data/manifests

ENV PATH="/app/backend/.venv/bin:$PATH" \
    MEZVUAT_DATA_DIR=/data \
    MEZVUAT_CORPUS_MANIFEST=/app/data/manifests/demo_corpus.json

USER mezvuat
EXPOSE 8000
CMD ["uvicorn", "mezvuat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
