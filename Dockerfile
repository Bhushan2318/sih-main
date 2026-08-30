# syntax=docker/dockerfile:1
#
# Sanket - one image, one process: FastAPI serves both the API and the built dashboard.
#
# The alternative (two Render services, or a static site calling a separate API) needs CORS
# and burns two free instances for one product; one origin needs neither.
#
# What this image deliberately does NOT do is train. A full retrain peaks around 2 GB and
# the free tier gives 512 MB, so the model is built on a 16 GB GitHub Actions runner and
# pulled in below as a release asset. See .github/workflows/refresh-data.yml.

# ---------------------------------------------------------------- stage 1: the dashboard
FROM node:20-alpine AS web
WORKDIR /web

# package files first so a source-only edit does not re-resolve the dependency tree
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Both are compile-time constants in the bundle, not runtime lookups:
#  - no API base => same-origin, because this image serves the API itself
#  - upload off  => the dropzone would trigger a retrain this box cannot survive
ENV VITE_ENABLE_UPLOAD=false
RUN npm run build


# ------------------------------------------------------------------- stage 2: the server
FROM python:3.12-slim AS app
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl for the release asset below; ca-certificates so HTTPS works at all
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Core requirements only. requirements-live.txt (eccodes/cfgrib/xarray) is for decoding
# fresh GRIB2 and is CI's job, not this container's - leaving it out keeps the image small
# and avoids the eccodes-on-slim mess entirely.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
# main.py mounts this directory only if it exists, so local dev (where it does not) is
# untouched and keeps using the Vite dev server.
COPY --from=web /web/dist ./app/static

# The trained model, the canonical parquet store and the metadata db are build inputs
# rather than source: they are regenerated every cycle and would otherwise be a 24 MB
# commit each time. CI publishes them to a fixed release tag; this pulls the latest.
#
# A missing or unreachable asset is NOT a build failure. The app already has an honest
# answer for having no model - /api/model/status reports model_trained: false and the UI
# renders its empty state with the reason - and that is a far better outcome than an image
# that refuses to build the night before a demo.
ARG DATA_ASSET_URL="https://github.com/Bhushan2318/sih-main/releases/download/data-latest/sanket-data.tar.gz"
RUN if [ -n "$DATA_ASSET_URL" ]; then \
      echo "fetching model+data: $DATA_ASSET_URL"; \
      if curl -fsSL --retry 3 --retry-delay 2 -o /tmp/data.tar.gz "$DATA_ASSET_URL"; then \
        tar -xzf /tmp/data.tar.gz -C /app && rm /tmp/data.tar.gz && \
        echo "unpacked:" && ls -la /app/data/models 2>/dev/null | head; \
      else \
        echo "WARNING: no data asset available; starting without a trained model"; \
      fi; \
    fi

# Render injects $PORT and expects the process to bind it on 0.0.0.0. The shell form is
# required here so the variable is actually expanded.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
