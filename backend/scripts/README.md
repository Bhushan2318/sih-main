# Scripts

Command-line tools for fetching data, building geography, training, evaluating and
deploying. Run them from `backend/` with the virtualenv active, as
`python -m scripts.<name>` or `python scripts/<name>.py`. Each script's docstring is its
full documentation: what it does, why it exists, and how it was measured.

## Index

**Fetching data.** Every fetch is idempotent and resumable, and pulls from a public archive
once.

| Script | What it does |
|---|---|
| `fetch_gefs_reforecast_sample.py` | Forecasts from the NOAA GEFSv12 reforecast: 666-district area means for eight surface variables, plus the gridded fields the convolutional model reads, from one download |
| `fetch_era5_cds_district_observations.py` | District observations from ERA5, in bulk from the Copernicus Climate Data Store |
| `fetch_era5_district_observations.py` | District observations from ERA5 through Open-Meteo: practical for recent days, while whole years come from the CDS script |
| `fetch_era5_observations.py` | The city-point ERA5 sample, through Open-Meteo |
| `fetch_imd_district_rainfall.py` | District rainfall from IMD's gauge-based gridded product, merged into an ERA5 observation file |
| `fetch_mjo_index.py` | The daily MJO index (NOAA PSL's OMI) |
| `fetch_grid_elevation.py` | Elevation for every 0.25° cell the district weight table uses |

**Geography**

| Script | What it does |
|---|---|
| `build_district_geo.py` | The 666 districts and their grid weight table, from GADM admin-2 boundaries |
| `build_claimed_territory_geo.py` | The map overlay for areas India claims but does not administer |
| `build_district_descriptors.py` | Per-district location, area, border distance and elevation, from geometry already on disk |
| `restore_island_rings.py` | Puts back the island outlines that simplifying the map's TopoJSON drops |

**Ingesting**

| Script | What it does |
|---|---|
| `ingest_districts_chunked.py` | Ingests a district-grain reforecast year in cycle-sized chunks |
| `ingest_backfill.py` | Loads backfilled reforecast years and their ERA5 labels into the canonical store |
| `fix_forecast_valid_date_offset.py` | One-off migration that corrected the one-day `valid_date` offset on rows ingested before 2026-08-29 |

**Training and evaluation**

| Script | What it does |
|---|---|
| `train_pooled.py` | Trains on any number of years without materialising them all at once |
| `run_pooled_overnight.py` | Runs a queue of pooled-training jobs unattended, so one failure does not lose the rest |
| `run_baselines.py` | Scores the baseline ladder against the trained classifier, on identical rows |
| `score_run_on_year.py` | Scores a saved run on a full calendar year, so runs trained on different years can be compared on the same one |
| `compare_verification_products.py` | Measures how far the ground truth disagrees with an independent reanalysis |
| `build_replay_cases.py` | Builds Replay's past events for one run |
| `measure_cnn_onnx_serving_memory.py` | Reproduces the CNN's ONNX serving memory and timing measurement |
| `ppt_figures.py` | Regenerates the deck's headline figures from the served model |

**Deploying**

| Script | What it does |
|---|---|
| `refresh_for_deploy.py` | Pulls the newest GEFS cycle, verifies what can be verified, retrains unless given `--skip-train`, and reports |
| `package_for_deploy.py` | Packs exactly what a serving box needs, including every prebuilt dashboard response |
| `publish_serving_model.py` | Publishes a finished run as the served model, through the promotion gate and an API check |
| `install_serving_model.py` | Installs a published serving model into a checkout and makes it current |

Files starting with `_` are subprocess workers that pooled training and the serving check
start themselves; they are not run by hand. `gen_frontend_region_codes.py` generated the
state-level map's region codes and has been unused since the map moved to districts.

## How the fetchers work

**Only the needed bytes.** For each `(variable, init, member)` GRIB2 file, only the
messages needed are downloaded, through HTTP `Range` requests keyed off the `.idx` sidecar,
not the whole 30–70 MB file. Messages for one lead day are contiguous, so they collapse
into one ranged request per lead day.

**Day *k*.** GEFS is 3-hourly. Lead day *k* is built from forecast hours ((k−1)·24, k·24]:
mean for state variables, sum for precipitation, vector mean for wind. It is labelled
`valid_date = init + (k − 1)`, because for a 00 UTC initialisation those hours fall on that
calendar day. Labelling it `init + k` once verified every forecast against the next day's
observation; do not change it. ERA5 is aggregated the same way per UTC calendar day.

**Lead-time coverage differs by variable** in the reforecast archive, and is reported at
the end of each run:

- temperature, humidity, rainfall, pressure, atmospheric moisture: Day 1–10
- 10 m wind: Day 1–5 only
- soil moisture: about Day 1–3 only

**Provenance.** Every forecast value records the GRIB2 file and message numbers it came
from, so any value can be traced back and re-derived.

### Canonical variables

| canonical column | GEFS source | ERA5 source (Open-Meteo name) | unit | notes |
|---|---|---|---|---|
| `t2m_c` | `tmp_2m` | `temperature_2m` | °C | K → °C |
| `rh2m_pct` | derived from `spfh_2m` + `tmp_2m` + `pres_sfc` | `relative_humidity_2m` | % | Bolton (1980) `es`; standard q → e inversion |
| `apcp_mm` / `precip_mm` | `apcp_sfc` | `precipitation` | mm | GEFS: 24 h total from 3 and 6 h buckets. ERA5: daily sum |
| `mslp_hpa` | `pres_msl` | `pressure_msl` | hPa | Pa → hPa |
| `psfc_hpa` | `pres_sfc` | `surface_pressure` | hPa | lower at altitude (Leh is about 660 hPa) |
| `pwat_kgm2` | `pwat_eatm` | `total_column_integrated_water_vapour` | kg/m² | atmospheric moisture |
| `wspd10m_ms`, `wdir10m_deg` | `ugrd_hgt` / `vgrd_hgt` (10 m) | `wind_speed_10m` / `wind_direction_10m` | m/s, ° | from daily-mean u and v; a scalar mean of degrees is wrong across 0/360 |
| `soilw_vol_pct` / `soil_moisture_pct` | `soilw_bgrnd` (0–0.1 m) | `soil_moisture_0_to_7cm` | % volumetric | both ×100 (m³/m³ → %) |

## The checked-in sample

`data/samples/gefs_reforecast_india_2019.parquet` and
`data/samples/era5_observations_india_2019.parquet` are a small real slice: 17 GEFSv12
reforecast cycles from 2019 at 36 city points (`india_cities.json`), with their ERA5
observations. They predate the move to districts and are kept because they are small
enough to commit, so CI and the README's Quick start can ingest and train on real data
without a download. The served model is trained on the full district archive instead.

## Licensing

- **GEFSv12 reforecast:** a work of the U.S. Government (NOAA), public domain, hosted on
  the AWS Open Data registry (`noaa-gefs-retrospective`).
- **ERA5:** Copernicus Climate Change Service information, CC-BY 4.0, from the Climate Data
  Store or redistributed by Open-Meteo. Attribute both if you publish data derived from it.
