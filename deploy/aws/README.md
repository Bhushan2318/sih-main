# Serving Sanket from an AWS EC2 instance

Replaces the Render free-tier plan (512 MB ceiling) with a paid EC2 box, chosen
deliberately over the free-tier Oracle path discussed earlier - this trades the "$0
hosting" framing in CLAUDE.md for headroom and no time-limited free-tier clock. Worth
updating CLAUDE.md's cost claim to match once this is live, so the docs and the actual
deployment do not disagree.

The image is unchanged: the same `Dockerfile` Render builds runs here as-is, because it
already does everything a serving box needs (fetches its own data/model artifact at
startup, no training, no live NOAA pull - see the Dockerfile's own comments). Moving host
never means moving architecture.

Render stays live until this is verified - nothing here touches `render.yaml`,
`RENDER_DEPLOY_HOOK`, or `.github/workflows/refresh-data.yml`'s existing deploy step.

## One-time instance bootstrap

On a fresh Ubuntu 22.04 EC2 instance (t3.small recommended - 2 vCPU/2 GB, comfortably
above the ~388 MB the app actually uses, at roughly $15/month on-demand):

```bash
ssh -i sanket-key.pem ubuntu@<instance-public-ip>
curl -fsSL https://raw.githubusercontent.com/Bhushan2318/sih-main/feature/cds-district-observations/deploy/aws/setup.sh | bash
```

Or copy `setup.sh` up with `scp` and run it directly. It:
1. Installs Docker (official convenience script), adds `ubuntu` to the `docker` group.
2. Clones this repo to `/opt/sanket` on `feature/cds-district-observations`
   (switch to `main` once this migration is merged there).
3. Builds the image from the repo's own `Dockerfile` - identical build Render runs.
4. Runs the container on port 8000, `--restart=always`, named `sanket`.
5. Installs `nginx` as a plain HTTP reverse proxy from port 80 to 8000 (no TLS yet - add
   a domain and `certbot` once one exists; an IP-only site cannot get a Let's Encrypt
   cert).

## Redeploying after a data refresh or a code change

```bash
/opt/sanket/deploy/aws/redeploy.sh
```

Pulls latest code, rebuilds, swaps the container. For a pure data refresh (new
`sanket-data.tar.gz` release asset, no code change), `docker restart sanket` alone is
enough and cheaper - the container's own `docker-entrypoint.sh` re-fetches the release
asset on every start regardless of image age.

## Verifying before cutover

```bash
curl -s http://<instance-public-ip>/api/health
```

Compare against Render's `/api/health` before touching DNS or removing the Render
service.

## Cost tracking

At $100 budget and ~$15/month for a t3.small, that is roughly 6-7 months of runway.
Set a AWS Budgets alert (Billing → Budgets → Create budget) at, say, $80 so there is
warning before the $100 runs out - not done automatically here since it needs your AWS
account, not something scriptable from outside it.

## Wiring CI to redeploy automatically (not done yet)

`refresh-data.yml` triggers Render via `RENDER_DEPLOY_HOOK` (a webhook secret). The AWS
equivalent needs an SSH-based step instead (e.g. `appleboy/ssh-action` with a deploy key
added as a GitHub secret), added *alongside* the existing Render step, not replacing it,
until AWS is confirmed serving correctly. That is a `.github/workflows` change and needs
explicit sign-off before it lands, per the working agreement - flagged here rather than
done speculatively.
