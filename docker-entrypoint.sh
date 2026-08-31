#!/bin/sh
# Fetch the current model + data, then serve.
#
# This deliberately happens at START, not at build. The artifact is a *rolling* release
# asset - CI replaces it every six hours - but its URL never changes, and Docker caches a
# RUN layer on the command string alone. So a build whose code has not changed reuses the
# cached download and ships whatever data was current when that layer was first built.
# Every refresh that only changed data would leave the site serving stale forecasts while
# reporting a successful deploy.
#
# Fetching here also means a plain restart picks up fresh data, with no rebuild at all.
#
# A failed fetch is not fatal: the build bakes a fallback copy (stale, but real), and if
# even that is missing the app reports model_trained: false honestly. Refusing to start
# would turn a GitHub outage into a site outage.
set -e

if [ -n "$DATA_ASSET_URL" ]; then
  echo "fetching model+data: $DATA_ASSET_URL"
  if curl -fsSL --retry 3 --retry-delay 2 -o /tmp/data.tar.gz "$DATA_ASSET_URL"; then
    tar -xzf /tmp/data.tar.gz -C /app
    rm -f /tmp/data.tar.gz
    echo "model+data refreshed at startup"
  else
    echo "WARNING: could not fetch the data asset; using whatever the image already has"
  fi
else
  echo "DATA_ASSET_URL unset; serving whatever the image already has"
fi

# exec so uvicorn is PID 1 and receives the platform's stop signals directly
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
