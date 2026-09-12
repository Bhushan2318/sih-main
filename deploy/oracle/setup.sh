#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 22.04 OCI Always Free VM. Idempotent: safe to
# re-run (docker install and repo clone both skip if already present).
set -euo pipefail

REPO_URL="https://github.com/Bhushan2318/sih-main.git"
BRANCH="feature/cds-district-observations"
APP_DIR="/opt/sanket"
PORT=8000

echo "== Docker =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "Docker installed. You may need to log out/in for group membership to apply."
else
  echo "Docker already installed, skipping."
fi

echo "== Repo =="
if [ ! -d "$APP_DIR/.git" ]; then
  sudo mkdir -p "$APP_DIR"
  sudo chown "$USER":"$USER" "$APP_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  echo "$APP_DIR already a git checkout, skipping clone."
fi

echo "== Build =="
cd "$APP_DIR"
sudo docker build -t sanket:latest .

echo "== Run =="
sudo docker rm -f sanket 2>/dev/null || true
sudo docker run -d --name sanket --restart=always -p "${PORT}:8000" sanket:latest

echo "== nginx (plain HTTP reverse proxy, no TLS yet) =="
if ! command -v nginx >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y nginx
fi
sudo tee /etc/nginx/sites-available/sanket >/dev/null <<NGINX
server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/sanket /etc/nginx/sites-enabled/sanket
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
sudo systemctl enable nginx

echo ""
echo "Done. Verify with:  curl -s http://localhost/api/health"
echo "From outside:       curl -s http://<this-vm-public-ip>/api/health"
