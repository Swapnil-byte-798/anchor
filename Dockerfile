# =============================================================================
# ANCHOR — container image.
#
# Zero keys, zero network at runtime. The only network access in this build is
# pip resolving the dependencies in pyproject.toml; NO model is ever downloaded,
# at build time or at start-up, because the default embedder is a deterministic
# hashing function and the default LLM is a local extractive stub.
#
#   docker build -t anchor .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" anchor
#
# On start the container ingests the corpus if data/index.db is absent, then
# serves the API on :8000. Nothing else is required to get a working service.
# =============================================================================

FROM python:3.11-slim

# Fail fast, no .pyc clutter, unbuffered logs so `docker logs` is live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANCHOR_CONFIG_FILE=config/default.yaml

# Non-root from the start. /app/data is created and chowned here so the index is
# writable even when no volume is mounted. (With a BIND mount the host
# directory's ownership wins, so create ./data yourself before `compose up` —
# see docker-compose.yml.)
RUN useradd --create-home --shell /bin/bash --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app

WORKDIR /app

# Dependency layer first: pyproject + the package source only, so editing the
# corpus or the config does not invalidate the pip cache layer.
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
RUN pip install --no-cache-dir -e .

# Everything the running service actually reads.
COPY --chown=app:app config/ ./config/
COPY --chown=app:app corpus/ ./corpus/
COPY --chown=app:app eval/ ./eval/

USER app

EXPOSE 8000

# urllib, not curl: the slim base has no curl and adding one would mean another
# apt layer for a three-line health probe.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# One sh -c: make the data dir, ingest only if the index is absent (so a mounted
# volume survives restarts), then hand PID 1 to uvicorn via exec.
CMD ["sh", "-c", "mkdir -p /app/data && if [ ! -f /app/data/index.db ]; then anchor ingest; fi && exec uvicorn anchor.api:create_app --factory --host 0.0.0.0 --port 8000"]
