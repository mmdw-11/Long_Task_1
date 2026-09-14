# Production image for the FastAPI service.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# A small set of system libraries keeps common Python wheels and document
# parsers working on Debian-based images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

COPY engine ./engine
COPY configs ./configs
COPY ollama_modelfiles ./ollama_modelfiles
# The submitted local BGE-M3 encoder and its trained route classifier.  These
# are copied from the verified experiment artifact, not downloaded at runtime.
COPY runs/router_learning/final_bge_m3_contrastive_smoke/encoder /app/model_assets/bge-m3
COPY runs/router_learning/final_bge_m3_contrastive_smoke/router /app/model_assets/bge-router

# Runtime state is deliberately outside the image layer and is mounted by
# docker-compose.yml at /app/runs.
RUN mkdir -p /app/runs

EXPOSE 8000

CMD ["uvicorn", "engine.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
