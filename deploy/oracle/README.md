# Serving Sanket from an Oracle Cloud Always Free VM

Why this exists: Render's free tier caps the serving box at 512 MB. An OCI Always Free
Ampere A1 VM gives up to 4 OCPU / 24 GB RAM, permanently, at $0 - the same "free tier
only" constraint in CLAUDE.md, just a roomier box. The image is unchanged: the same
`Dockerfile` that Render builds runs here, because it already does everything a serving
box needs (fetches its own data/model artifact at startup, no training, no live NOAA
pull - see the Dockerfile's own comments). Moving host does not mean moving architecture.

Render stays live during this migration. Nothing here touches `render.yaml`,
`RENDER_DEPLOY_HOOK`, or `.github/workflows/refresh-data.yml`'s existing deploy step -
this is a parallel target to verify before anything is cut over or removed.

## One-time VM bootstrap

On a fresh Ubuntu 22.04 Always Free instance, as the `ubuntu` user:

```bash
curl -fsSL https://raw.githubusercontent.com/Bhushan2318/sih-main/feature/cds-district-observations/deploy/oracle/setup.sh | bash
```

Or copy `setup.sh` up and run it directly. It:
1. Installs Docker (official convenience script) and adds `ubuntu` to the `docker` group.
2. Clones this repo to `/opt/sanket` on the `feature/cds-district-observations` branch
   (change to `main` once this migration is merged there).
3. Builds the image from the repo's own `Dockerfile` - same build Render runs, done
   locally on the VM so no external registry or cross-arch build is needed (the VM is
   arm64, and both `python:3.12-slim` and `node:20-alpine` publish arm64 images, so this
   is a native build, not an emulated one).
4. Runs the container on port 8000, restart-always, named `sanket`.
5. Installs `nginx` as a plain reverse proxy from port 80 to 8000 (no TLS yet - add a
   domain and `certbot` once one exists; an IP-only site cannot get a Let's Encrypt cert).

## Redeploying after a data refresh

`refresh-data.yml` publishes a new `sanket-data.tar.gz` release asset every 6 hours (or
on a new commit to the serving image). The container's own `docker-entrypoint.sh` already
re-fetches that asset on every start - so redeploying is just restarting it:

```bash
/opt/sanket/deploy/oracle/redeploy.sh
```

Which pulls the latest code (in case the Dockerfile or app changed), rebuilds, and swaps
the container. For a pure data refresh with no code change, `docker restart sanket` alone
is enough and cheaper - the entrypoint re-fetches the release asset regardless.

## Verifying before cutover

```bash
curl -s http://<vm-public-ip>/api/health
```

Should return the same shape Render's `/api/health` does. Compare the two side by side
before touching DNS, `SITE_URL`, or removing the Render service.

## Wiring CI to redeploy automatically (not done yet)

`refresh-data.yml` triggers Render via `RENDER_DEPLOY_HOOK` (a webhook secret). The
Oracle equivalent needs an SSH-based step instead (e.g. `appleboy/ssh-action` with a
deploy key added as a GitHub secret), added *alongside* the existing Render step, not
replacing it, until Oracle is confirmed serving correctly. That is a `.github/workflows`
change and needs explicit sign-off before it lands, per the working agreement - flagged
here rather than done speculatively.
