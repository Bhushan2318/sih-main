#!/usr/bin/env bash
# Pull latest code, rebuild the image, and swap the running container.
#
# For a pure data refresh (new sanket-data.tar.gz release asset, no code change),
# `docker restart sanket` alone is enough and cheaper - docker-entrypoint.sh re-fetches
# the release asset on every start regardless of image age. This script is for when the
# Dockerfile, backend, or frontend source actually changed.
set -euo pipefail

APP_DIR="/opt/sanket"
PORT=8000

cd "$APP_DIR"
git fetch origin
git reset --hard "origin/$(git rev-parse --abbrev-ref HEAD)"

docker build -t sanket:latest .
docker rm -f sanket 2>/dev/null || true
docker run -d --name sanket --restart=always -p "${PORT}:8000" sanket:latest

echo "Redeployed. Verify with: curl -s http://localhost/api/health"
