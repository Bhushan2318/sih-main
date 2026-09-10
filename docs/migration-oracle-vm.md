# Migrating serving to an Oracle Always Free VM

Written 2026-09-10, while the district ingest was still running. Nothing here has been
executed yet — this is the plan, with the risky parts measured rather than assumed.

## Why

Serving on Render free is at the wall. Measured 2026-09-10 against the 661-district
store:

```
/api/regions   729 MB
/api/alerts    740 MB      ceiling 512 MB, killed rather than throttled
```

The 388–442 MB figures in the older docs were measured at 36 districts. A cycle went from
11,325 rows to 221,055, and the region panel reads per cycle. 666 districts do not fit on
Render free, and no amount of tuning closes a 230 MB overshoot.

An Always Free Ampere VM has 12 GB. That is 24x the headroom, and it is still $0, so
rule 6 survives — arguably strengthened, since "666 districts on free infrastructure" is
a better line than "36".

## Know this before planning around it

**Oracle halved the Always Free Ampere allowance on 15 June 2026**, from 4 OCPU / 24 GB to
**2 OCPU / 12 GB**, with no public announcement, and told users instances above the new
limit would be terminated from 18 August 2026. Plan for 12 GB.

**Ampere capacity is frequently unavailable.** "Out of capacity" on instance creation is
the normal experience in popular regions, not the exception. Confirm you can actually
create the instance before scheduling any work around it.

**Keep Render running until the VM is proven end to end.** Oracle changed the rules and
terminated instances once already this year, and the finale is 6 December. Two homes until
one is demonstrably better.

## The one real technical risk, and it is cleared

Ampere is ARM64; the current image is x86. Every serving dependency was checked for an
aarch64 cp312 wheel on 2026-09-10 — all present, no source builds:

```
numpy 1.26.4   pandas 2.1.4   pyarrow 16.1.0   xgboost 2.0.3   scikit-learn 1.4.2
shap 0.45.1    shapely 2.0.7  rapidfuzz 3.13.0 uvloop 0.22.1    httptools 0.8.0
```

`python:3.12-slim` and `node:20-alpine` both publish arm64 variants, so the existing
two-stage Dockerfile should build unchanged under `--platform linux/arm64`.

## Steps

### 1. Create the instance
Ampere A1, 2 OCPU / 12 GB, Ubuntu LTS, in a region you can actually get capacity in.
Boot volume plus block volume within the 200 GB Always Free total. Add your SSH key at
creation; there is no password login.

### 2. Open the port twice — this is where people lose an afternoon
Oracle requires **both**:
- an ingress rule in the subnet's **security list** (VCN networking), and
- a rule in the instance's **own iptables**, because Oracle's Ubuntu images ship with a
  restrictive default `INPUT` chain.

Opening only the security list produces a port that looks open in the console and refuses
connections. Persist the iptables rule (`netfilter-persistent save`) or it dies on reboot.

### 3. Build for ARM64
Do **not** cross-build on the dev Mac: it is Intel x86_64, so `buildx` would run the npm
and pip stages under QEMU emulation, which is slow enough to be painful.

Two better options:
- **Build on the VM.** Native arm64, 2 OCPU. Slow for `npm ci` but correct, and needs no
  new infrastructure.
- **Build in GitHub Actions on an arm64 runner** and push to a registry. Native speed, and
  free for public repos. This means a workflow change, which CLAUDE.md rule 8 says must be
  called out explicitly — so raise it before doing it.

### 4. Put the data on the VM, not in a Release asset
The entrypoint currently pulls a rolling release asset from `DATA_ASSET_URL`, and
GitHub caps a release asset at 2 GB. The district store is heading for ~5 GB (snappy) and
the gridded fields are another ~1.5 GB, so that path is already broken by size.

With 200 GB of block storage, mount a volume and keep the canonical store on it directly.
Then `DATA_ASSET_URL` becomes optional rather than load-bearing, and a restart stops
depending on GitHub being up. Keep the entrypoint's fallback behaviour: a failed fetch
must not be fatal, and a missing model must still report `model_trained: false` honestly.

### 5. TLS and a front door
Render terminated SSL for you; the VM will not. Caddy is the least effort — it obtains and
renews Let's Encrypt certificates automatically and reverse-proxies to the container.
You need a DNS name pointed at the VM's public IP first.

### 6. Deploy loop
Simplest that works: `docker compose pull && docker compose up -d` behind a small script,
triggered manually or by a GitHub Actions SSH step. **Do not** wire an automatic deploy on
push — `autoDeploy: false` was set on Render on purpose, and that reasoning carries over.

### 7. Verify before switching DNS
- `/api/health` returns 200 and reports `memory_mb`
- `/api/regions` returns the full district set, and note its RSS — it should be a small
  fraction of 12 GB where it was 142% of 512 MB
- `/api/model/status` reports the expected run
- websockets actually work, which they never did on Render free. The client currently
  falls back to 60 s polling and reports the socket as closed; that fallback can stay, but
  the live path becomes real.

## What this invalidates in the docs

`CLAUDE.md`'s architecture section is load-bearing on the 512 MB ceiling — "the serving
box is **killed**, not throttled, at 512 MB. That is why the split exists." Several
measured facts hang off it: the +253 MB cycle-listing fix, the CNN's +51 MB onnxruntime
budget, "serving currently uses 388 MB of 512".

Do not delete those. They are true of Render and they are the reason the code is shaped as
it is. Rewrite the section to say which platform each figure describes, and re-measure on
the VM before quoting any new number.

## What stops being urgent

The categorical-read fix for `parquet_store` — Arrow holds a cycle in 18 MB and
`to_pandas()` inflates it to 84 MB by making Python objects of every string. It is a real
4x win and was attempted on 2026-09-10, but it changes pandas groupby semantics
project-wide: a groupby on categorical columns returns the cartesian product of categories
unless `observed=True`, which broke `build_training_frame` immediately (24,928 rows became
90,032). It was reverted.

At 512 MB that fix was mandatory. At 12 GB it is hygiene, and auditing `observed=True`
across the feature code three months before the finale is not a trade worth making.
Revisit after December.
