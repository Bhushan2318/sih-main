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
  rm -f /tmp/data.tar.gz
  if curl -fsSL --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 2 \
    -o /tmp/data.tar.gz "$DATA_ASSET_URL"; then
    # Validate before clearing the fallback. A malformed or path-unsafe release must not
    # replace the known-good copy baked into the image.
    if python /usr/local/bin/validate-release-archive /tmp/data.tar.gz \
      --allow-prefix data/ --allow-file metadata.db \
      --require data/models/current.json \
      --require-prefix data/canonical/ \
      --require-prefix data/geo/ \
      --require data/summary.json \
      --require metadata.db; then
      # Clear the canonical store before extracting, and only once validation has
      # succeeded. tar overwrites what the archive contains but never removes what it does
      # not, and packaging now COMPACTS the store - dropping whole batch_id= partitions
      # whose rows were entirely superseded. Extracting a compacted store over an older one
      # therefore leaves orphan partitions behind. store_signature() enumerates every
      # partition on disk, so it stops matching the summary.json shipped beside it, and the
      # Model page falls back to reporting the store as unavailable. The orphans are also
      # still read, quietly undoing part of what compaction bought.
      #
      # After validation, never before: deleting first would turn a GitHub outage into an
      # empty store, and the fallback baked into the image exists precisely to prevent that.
      rm -rf /app/data/canonical /app/data/models /app/data/geo /app/data/analysis \
        /app/data/summary.json /app/metadata.db
      if ! tar --extract --gzip --file /tmp/data.tar.gz --directory /app \
        --no-same-owner --no-same-permissions; then
        echo "ERROR: model+data archive passed validation but extraction failed; refusing to start" >&2
        rm -f /tmp/data.tar.gz
        exit 1
      fi
      test -d /app/data/canonical
      test -d /app/data/models
      test -f /app/data/models/current.json
      test -f /app/data/summary.json
      test -f /app/metadata.db
      rm -f /tmp/data.tar.gz
      echo "model+data refreshed at startup"
    else
      echo "ERROR: downloaded model+data archive was rejected; using the image fallback" >&2
      rm -f /tmp/data.tar.gz
    fi
  else
    rm -f /tmp/data.tar.gz
    echo "WARNING: could not fetch the data asset; using whatever the image already has"
  fi
else
  echo "DATA_ASSET_URL unset; serving whatever the image already has"
fi

# exec so uvicorn is PID 1 and receives the platform's stop signals directly
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
