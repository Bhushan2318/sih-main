"""Fetch one operational GEFS cycle for the Indian city points.

Source
------
NOAA GEFS operational, 0.25 degree ``pgrb2sp25`` product (US Government work, public
domain), issued 4x daily at 00/06/12/18 UTC out to +240 h in 3-hourly steps.

Two transports, same decode path:

  * **NOMADS grib-filter** (preferred) subsets to an India bounding box server-side:
    ~0.19 MB per step against ~4.4 MB of byte-ranged global messages, and one request
    instead of nine. NOMADS only retains roughly the last three days.
  * **AWS S3** ``noaa-gefs-pds`` (fallback) for anything older, using HTTP ``Range``
    requests keyed off each file's ``.idx`` sidecar.

Output is a wide CSV whose measurement columns match
``data/samples/gefs_reforecast_india_2019.csv`` exactly, so live cycles land in the same
canonical shape as the training data.

Two consistency rules that must not be relaxed casually
-------------------------------------------------------
1. **0.25 degree only.** The 0.50 degree product was measured against this one at the 30
   city points and differs by up to 6.1 C in 2 m temperature and 10 %RH - worst in exactly
   the mountain regions (Sikkim, Himachal, Ladakh) the model flags most often. The
   regressors were trained on 0.25 degree reforecast data; feeding them 0.50 degree values
   would shift their inputs relative to training.

2. **Exactly five members.** Operational GEFS carries 31 (``gec00`` + ``gep01``-``gep30``);
   the reforecast the models were trained on carries five. Ensemble spread grows with
   member count, and ``spread_mean``/``spread_max`` are classifier inputs - scoring a
   31-member spread with a 5-member-trained model would bias its most important features.
   Widening this set requires retraining from scratch on matching data.

Relative humidity is read directly here, whereas the training fetch derived it from
specific humidity via Bolton (1980). Measured at the same 30 points on the same cycle,
the two agree to within 0.50 %RH (mean -0.03), which is far below the models' error scale.
"""

from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")  # cfgrib / urllib3 chatter on decode

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("forecastguard.live.gefs")

S3_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
NOMADS_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

BACKEND_DIR = Path(__file__).resolve().parents[2]
CITIES_JSON = BACKEND_DIR / "scripts" / "india_cities.json"

# India bounding box for the NOMADS subregion filter. Generous enough to contain every
# city point plus a margin for nearest-grid-point selection at the edges.
BBOX = {"toplat": 38, "bottomlat": 6, "leftlon": 68, "rightlon": 98}

LEAD_DAYS = list(range(1, 11))
STEP_HOURS = 3
MAX_LEAD_H = 240

# canonical output column -> how to read and reduce it.
#   short_name : cfgrib key after decode
#   grib       : (abbrev, level substring) for locating the message in an .idx
#   agg        : daily reduction over the samples inside a lead day
VAR_SPEC: dict[str, dict] = {
    "t2m_c":         {"short_name": "t2m",   "grib": ("TMP", "2 m above ground"),   "agg": "mean"},
    "rh2m_pct":      {"short_name": "r2",    "grib": ("RH", "2 m above ground"),    "agg": "mean"},
    "psfc_hpa":      {"short_name": "sp",    "grib": ("PRES", "surface"),           "agg": "mean"},
    "mslp_hpa":      {"short_name": "prmsl", "grib": ("PRMSL", "mean sea level"),   "agg": "mean"},
    "pwat_kgm2":     {"short_name": "pwat",  "grib": ("PWAT", "entire atmosphere"), "agg": "mean"},
    "soilw_vol_pct": {"short_name": "soilw", "grib": ("SOILW", "0-0.1 m below"),    "agg": "mean"},
    "u10":           {"short_name": "u10",   "grib": ("UGRD", "10 m above ground"), "agg": "mean"},
    "v10":           {"short_name": "v10",   "grib": ("VGRD", "10 m above ground"), "agg": "mean"},
    # APCP buckets RESET EVERY 6 HOURS and the 3-hourly steps overlap: f003 is "0-3 hour
    # acc", f006 is "0-6 hour acc". Summing every 3-hourly value would roughly double the
    # daily total. Only the fh % 6 == 0 steps are used, and those tile the day exactly.
    "apcp_mm":       {"short_name": "tp",    "grib": ("APCP", "surface"),
                      "agg": "sum", "step_multiple": 6},
}

# Variables aggregated by SUMMING the day's steps rather than averaging them. They are
# singled out because a short sum is wrong in a way a short mean is not: a mean over 6 of
# 8 samples is the same quantity measured more noisily, while a sum over 2 of 4 is roughly
# half the real total. apcp_mm is precipitation - the variable behind most busts - so a
# short sum understates risk instead of blurring it.
SUM_VARIABLES = frozenset(c for c, spec in VAR_SPEC.items() if spec.get("agg") == "sum")

NOMADS_VARS = ["TMP", "RH", "PRES", "PRMSL", "PWAT", "APCP", "UGRD", "VGRD", "SOILW"]

HTTP_RETRIES = 5
HTTP_BACKOFF = 2.0

# NOMADS is a shared public service that asks callers to stay modest; pushing 20 parallel
# requests at it earns a 302 to an error page rather than data. Six workers keeps us near
# 100 requests/minute, which it serves without complaint. S3 has no such limit.
NOMADS_MAX_WORKERS = 6
# 302 (redirect to NOMADS' throttle page), 429 and 5xx are transient - back off and retry.
# 403/404 mean the cycle genuinely is not there, which is a different decision.
RETRYABLE_STATUS = {302, 429, 500, 502, 503, 504}

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/phase6-live (SIH 2026; NCMRWF PS 26079)"
# Default pool is 10; without this the extra workers churn connections and log warnings.
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


class CycleUnavailable(RuntimeError):
    """The cycle is not published (yet) at either transport."""


@dataclass
class FetchReport:
    init_date: date
    cycle_hour: str
    members: list
    transport: str = ""
    steps_expected: int = 0
    steps_fetched: int = 0
    bytes_downloaded: int = 0
    missing_steps: list = field(default_factory=list)
    # Steps NOMADS refused that were then fetched from S3 instead. They are already
    # counted in steps_fetched; this says how much of the pull leaned on the fallback.
    steps_recovered: int = 0
    seconds: float = 0.0
    # (member, lead_day, variable) groups that were reduced from fewer samples than a
    # complete day would give. A daily mean over 6 of 8 samples is a slightly different
    # quantity from the one the models were trained on, so it is reported rather than
    # quietly accepted.
    undersampled: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.steps_fetched == self.steps_expected and not self.undersampled

    @property
    def undersampled_sums(self) -> list:
        """The undersampled groups that are accumulations, which are the damaging ones.

        Entries look like ``gep01/D2/apcp_mm(2/4)``; matching on ``/<var>(`` keys off the
        variable segment rather than a substring that could appear in a member name.
        """
        return [u for u in self.undersampled
                if any(f"/{var}(" in u for var in SUM_VARIABLES)]

    @property
    def step_completeness(self) -> float:
        """Fraction of expected GRIB steps actually fetched, 0.0-1.0."""
        if self.steps_expected <= 0:
            return 0.0
        return self.steps_fetched / self.steps_expected


# --------------------------------------------------------------------------------------
# transports
# --------------------------------------------------------------------------------------

def _get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    last: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(url, params=params, headers=headers, timeout=180)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (403, 404):
                raise FileNotFoundError(f"{r.status_code} for {r.url}")
            last = requests.HTTPError(f"{r.status_code} for {r.url}")
            if r.status_code not in RETRYABLE_STATUS:
                break
        except FileNotFoundError:
            raise
        except requests.RequestException as exc:
            last = exc
        # Exponential rather than linear: a throttled service needs real space, and the
        # earlier linear backoff kept re-arriving while the limit was still in force.
        time.sleep(HTTP_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} tries: {url} ({last})")


def _s3_key(init: date, hh: str, member: str, fh: int) -> str:
    return (f"gefs.{init:%Y%m%d}/{hh}/atmos/pgrb2sp25/"
            f"{member}.t{hh}z.pgrb2s.0p25.f{fh:03d}")


def _fetch_step_nomads(init: date, hh: str, member: str, fh: int) -> bytes:
    """One request returns every wanted variable, already cut to the India box."""
    params = {
        "dir": f"/gefs.{init:%Y%m%d}/{hh}/atmos/pgrb2sp25",
        "file": f"{member}.t{hh}z.pgrb2s.0p25.f{fh:03d}",
        "subregion": "",
        **{f"var_{v}": "on" for v in NOMADS_VARS},
        **{k: str(v) for k, v in BBOX.items()},
    }
    r = _get(NOMADS_URL, params=params)
    if not r.content.startswith(b"GRIB"):
        # NOMADS answers 200 with an HTML error page when a cycle has aged out.
        raise FileNotFoundError(f"NOMADS returned no GRIB for f{fh:03d} {member}")
    return r.content


def _parse_idx(text: str) -> list:
    lines = [l for l in text.splitlines() if l.strip()]
    starts = [int(l.split(":", 3)[1]) for l in lines]
    out = []
    for i, l in enumerate(lines):
        p = l.split(":")
        out.append({
            "start": int(p[1]),
            "end": starts[i + 1] if i + 1 < len(starts) else None,
            "abbrev": p[3],
            "level": p[4],
        })
    return out


def _fetch_step_s3(init: date, hh: str, member: str, fh: int) -> bytes:
    """Byte-range the wanted messages out of the global file (NOMADS fallback)."""
    key = _s3_key(init, hh, member, fh)
    recs = _parse_idx(_get(f"{S3_BUCKET}/{key}.idx").text)

    wanted = []
    for spec in VAR_SPEC.values():
        abbrev, level_sub = spec["grib"]
        hit = next((r for r in recs
                    if r["abbrev"] == abbrev and level_sub in r["level"]), None)
        if hit is not None:
            wanted.append(hit)

    # Collapse adjacent messages so contiguous ones cost a single request.
    wanted.sort(key=lambda r: r["start"])
    spans: list = []
    for r in wanted:
        if spans and spans[-1][1] == r["start"]:
            s, _ = spans.pop()
            spans.append((s, r["end"]))
        else:
            spans.append((r["start"], r["end"]))

    blobs = []
    for start, end in spans:
        rng = f"bytes={start}-{end - 1}" if end is not None else f"bytes={start}-"
        blobs.append(_get(f"{S3_BUCKET}/{key}", headers={"Range": rng}).content)
    return b"".join(blobs)


def _fetch_step(init: date, hh: str, member: str, fh: int, transport: str) -> bytes:
    if transport == "nomads":
        return _fetch_step_nomads(init, hh, member, fh)
    return _fetch_step_s3(init, hh, member, fh)


def choose_transport(init: date, hh: str, member: str = "gec00") -> str:
    """NOMADS when it still holds the cycle, S3 otherwise. Raises if neither has it."""
    try:
        _fetch_step_nomads(init, hh, member, 3)
        return "nomads"
    except (FileNotFoundError, RuntimeError) as exc:
        log.info("NOMADS unavailable for %s %sZ (%s); falling back to S3", init, hh, exc)
    try:
        _get(f"{S3_BUCKET}/{_s3_key(init, hh, member, 3)}.idx")
        return "s3"
    except Exception as exc:  # noqa: BLE001
        raise CycleUnavailable(f"cycle {init} {hh}Z not published at NOMADS or S3: {exc}")


# --------------------------------------------------------------------------------------
# decode + point extraction
# --------------------------------------------------------------------------------------

def _extract_points(blob: bytes, cities: pd.DataFrame, scratch: Path) -> dict:
    """Nearest-grid-point value per city for each variable in one step's GRIB blob."""
    import cfgrib  # imported lazily: only the live path needs eccodes present

    tmp = scratch / f".step_{time.time_ns()}_{id(blob) & 0xffff}.grib2"
    tmp.write_bytes(blob)
    try:
        datasets = cfgrib.open_datasets(str(tmp), backend_kwargs={"indexpath": ""})
        lats = cities["lat"].to_numpy()
        lons = cities["lon"].to_numpy()

        out: dict = {}
        for ds in datasets:
            import xarray as xr
            sel_lat = xr.DataArray(lats, dims="city")
            # GEFS longitudes run 0..360; the India box stays positive either way.
            sel_lon = xr.DataArray(lons % 360, dims="city")
            for col, spec in VAR_SPEC.items():
                sn = spec["short_name"]
                if sn not in ds.data_vars or col in out:
                    continue
                vals = ds[sn].sel(latitude=sel_lat, longitude=sel_lon, method="nearest")
                out[col] = np.asarray(vals.values, dtype=float).reshape(-1)
        return out
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------

def _lead_day_of(init_dt: datetime, fh: int) -> tuple:
    """(UTC calendar date, lead day) that forecast hour `fh` contributes to.

    Aggregation is by the valid time's calendar day, not by a window of hours since
    issue, because the observations these forecasts are verified against are calendar-day
    aggregates. For a 00 UTC cycle the two are identical - hours ((k-1)*24, k*24] are
    exactly calendar day init+k - so this reproduces the training pipeline byte for byte.
    For 06/12/18 UTC cycles a lead-hour window would straddle two calendar days and could
    never line up with its verifying observation.

    The subtraction puts a sample valid at midnight at the END of the preceding day rather
    than the start of the next one, matching how the accumulation buckets are labelled.
    """
    valid = init_dt + timedelta(hours=fh)
    day = (valid - timedelta(seconds=1)).date()
    # Lead day 1 is the first calendar day the run covers (the init day itself for a 00 UTC
    # cycle), so valid_date == init_date + (lead - 1). This is the convention the training
    # data now uses after the one-day labelling fix in fetch_gefs_reforecast_sample.py.
    return day, (day - init_dt.date()).days + 1


def _steps_by_day(init_dt: datetime, step_multiple: int = STEP_HOURS) -> dict:
    """{lead_day: [forecast hours]} for one sampling cadence."""
    out: dict = {}
    for fh in range(step_multiple, MAX_LEAD_H + 1, step_multiple):
        _, lead = _lead_day_of(init_dt, fh)
        out.setdefault(lead, []).append(fh)
    return out


def _expected_samples(step_multiple: int) -> int:
    """Samples a fully-covered day contains at this cadence (8 at 3-hourly, 4 at 6-hourly)."""
    return 24 // step_multiple


def _wind_speed_dir(u: np.ndarray, v: np.ndarray):
    spd = np.sqrt(u ** 2 + v ** 2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return spd, direction


def load_cities() -> pd.DataFrame:
    if not CITIES_JSON.exists():
        raise FileNotFoundError(f"missing city table: {CITIES_JSON}")
    return pd.DataFrame(pd.read_json(CITIES_JSON))


def _recover_steps_from_s3(
    init: date,
    cycle_hour: str,
    failed: list,
    cities: pd.DataFrame,
    scratch: Path,
    step_values: dict,
    report: FetchReport,
    workers: int,
) -> list:
    """Re-fetch the steps NOMADS would not serve from S3, which is not rate-limited.

    NOMADS answers a burst it considers too fast with a 302 to its throttle page, and the
    windows last longer than the retry ladder in ``_get`` - so a mid-pull throttle leaves a
    contiguous block of holes that retrying NOMADS cannot fill. S3 carries the same
    ``pgrb2sp25`` files, so the holes are refilled there before the daily reduction runs.

    ``step_values`` and ``report`` are updated in place; the recovered step ids are
    returned. Anything S3 also cannot supply stays listed in ``report.missing_steps``
    exactly as the main pass left it, so a genuinely thin cycle is still reported as thin.
    """
    def job(member: str, fh: int):
        blob = _fetch_step_s3(init, cycle_hour, member, fh)
        return member, fh, len(blob), _extract_points(blob, cities, scratch)

    recovered = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(failed)))) as pool:
        futures = {pool.submit(job, m, fh): (m, fh) for m, fh in failed}
        for fut in as_completed(futures):
            m, fh = futures[fut]
            try:
                member, hour, nbytes, values = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("S3 could not fill %s f%03d either: %s", m, fh, exc)
                continue
            step_values[(member, hour)] = values
            report.bytes_downloaded += nbytes
            report.steps_fetched += 1
            recovered.append(f"{member}/f{hour:03d}")

    if recovered:
        filled = set(recovered)
        report.missing_steps = [s for s in report.missing_steps if s not in filled]
        report.steps_recovered += len(recovered)
        log.info("S3 filled %d of %d gaps NOMADS left in %s %sZ",
                 len(recovered), len(failed), init, cycle_hour)
    return recovered


def fetch_cycle(
    init: date,
    cycle_hour: str,
    members: list,
    workers: int = 20,
    scratch: Optional[Path] = None,
    transport: Optional[str] = None,
) -> tuple:
    """Download and reduce one cycle. Returns (wide DataFrame, FetchReport)."""
    cities = load_cities()
    scratch = scratch or Path(BACKEND_DIR / "data" / "_live_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    transport = transport or choose_transport(init, cycle_hour, members[0])
    requested_workers = workers
    if transport == "nomads":
        workers = min(workers, NOMADS_MAX_WORKERS)
    all_steps = list(range(STEP_HOURS, MAX_LEAD_H + 1, STEP_HOURS))
    report = FetchReport(init_date=init, cycle_hour=cycle_hour, members=list(members),
                         transport=transport, steps_expected=len(all_steps) * len(members))

    t0 = time.time()
    # step_values[(member, fh)] = {column: array over cities}
    step_values: dict = {}

    def job(member: str, fh: int):
        blob = _fetch_step(init, cycle_hour, member, fh, transport)
        return member, fh, len(blob), _extract_points(blob, cities, scratch)

    failed: list = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, m, fh): (m, fh)
                   for m in members for fh in all_steps}
        for fut in as_completed(futures):
            m, fh = futures[fut]
            try:
                member, hour, nbytes, values = fut.result()
            except Exception as exc:  # noqa: BLE001
                # A missing step is recorded, not fatal: a cycle with a few gaps is still
                # usable and the gaps are reported rather than filled in.
                log.warning("step %s f%03d failed: %s", m, fh, exc)
                report.missing_steps.append(f"{m}/f{fh:03d}")
                failed.append((m, fh))
                continue
            step_values[(member, hour)] = values
            report.bytes_downloaded += nbytes
            report.steps_fetched += 1

    # Only NOMADS gaps are worth a second transport. If the pull was already on S3 there
    # is nowhere else to go, and a failure there means the step genuinely is not there.
    if transport == "nomads" and failed:
        _recover_steps_from_s3(init, cycle_hour, failed, cities, scratch, step_values,
                               report, requested_workers)

    report.seconds = time.time() - t0
    if not step_values:
        raise CycleUnavailable(f"no steps could be fetched for {init} {cycle_hour}Z")

    frame = _reduce_to_daily(step_values, cities, init, cycle_hour, members, transport,
                             report)
    return frame, report


def _reduce_to_daily(step_values: dict, cities: pd.DataFrame, init: date,
                     cycle_hour: str, members: list, transport: str,
                     report: Optional[FetchReport] = None) -> pd.DataFrame:
    """3-hourly point samples -> one wide row per (city, member, lead day)."""
    rows = []
    n_city = len(cities)
    init_dt = datetime(init.year, init.month, init.day, int(cycle_hour), tzinfo=timezone.utc)

    # cadence -> {lead_day: hours}. Instantaneous fields sample 3-hourly; accumulations
    # only use the non-overlapping 6-hourly buckets.
    by_cadence = {m: _steps_by_day(init_dt, m) for m in {STEP_HOURS, 6}}

    for member in members:
        for lead in LEAD_DAYS:
            record: dict = {}
            used_steps: dict = {}

            for col, spec in VAR_SPEC.items():
                step_mult = spec.get("step_multiple", STEP_HOURS)
                hours = by_cadence[step_mult].get(lead, [])
                # A day only partly inside the +240 h horizon (every lead 10 for a non-00Z
                # cycle) is dropped: a "daily mean" over part of a day is a different
                # quantity from the one the models were trained on.
                if len(hours) < _expected_samples(step_mult):
                    continue
                stack = [step_values[(member, h)][col]
                         for h in hours
                         if (member, h) in step_values and col in step_values[(member, h)]]
                if not stack:
                    continue
                if report is not None and len(stack) < len(hours):
                    report.undersampled.append(
                        f"{member}/D{lead}/{col}({len(stack)}/{len(hours)})"
                    )
                arr = np.vstack(stack)
                record[col] = arr.sum(axis=0) if spec["agg"] == "sum" else arr.mean(axis=0)
                used_steps[col] = hours

            if not record:
                continue

            block = pd.DataFrame({"city": cities["city"].to_numpy()})
            block["state"] = cities["state"].to_numpy()
            block["region"] = cities["region"].to_numpy()
            block["latitude"] = cities["lat"].to_numpy()
            block["longitude"] = cities["lon"].to_numpy()
            block["init_date"] = init_dt.date()
            block["valid_date"] = init_dt.date() + timedelta(days=lead - 1)
            block["lead_day"] = lead
            # Member ids are normalised to the reforecast's naming (gec00 -> c00) so live
            # rows sit in the same member namespace as the training data.
            block["member"] = member.replace("ge", "", 1)

            for col, values in record.items():
                block[col] = values[:n_city]

            # unit conversions to canonical, mirroring the training fetch exactly
            if "t2m_c" in block:
                block["t2m_c"] = block["t2m_c"] - 273.15
            if "mslp_hpa" in block:
                block["mslp_hpa"] = block["mslp_hpa"] / 100.0
            if "psfc_hpa" in block:
                block["psfc_hpa"] = block["psfc_hpa"] / 100.0
            if "soilw_vol_pct" in block:
                block["soilw_vol_pct"] = block["soilw_vol_pct"] * 100.0
            if {"u10", "v10"}.issubset(block.columns):
                spd, drc = _wind_speed_dir(block["u10"].to_numpy(), block["v10"].to_numpy())
                block["wspd10m_ms"], block["wdir10m_deg"] = spd, drc
                block = block.drop(columns=["u10", "v10"])

            src = (f"{'NOMADS grib-filter' if transport == 'nomads' else S3_BUCKET}"
                   f" gefs.{init:%Y%m%d}/{cycle_hour}/atmos/pgrb2sp25/{member}")
            for col in ("t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
                        "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct"):
                if col in block.columns:
                    block[f"src_{col}"] = src
                    key = "u10" if col.startswith("wspd") or col.startswith("wdir") else col
                    block[f"srcmsg_{col}"] = ",".join(
                        f"f{h:03d}" for h in used_steps.get(key, used_steps.get(col, []))
                    )
            rows.append(block)

    if not rows:
        raise CycleUnavailable("no lead days could be reduced from the fetched steps")

    out = pd.concat(rows, ignore_index=True)
    lead_col = ["city", "state", "region", "latitude", "longitude",
                "init_date", "valid_date", "lead_day", "member"]
    value_cols = [c for c in ["t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "psfc_hpa",
                              "pwat_kgm2", "wspd10m_ms", "wdir10m_deg", "soilw_vol_pct"]
                  if c in out.columns]
    src_cols = sorted(c for c in out.columns if c.startswith(("src_", "srcmsg_")))
    return out[lead_col + value_cols + src_cols].sort_values(
        ["city", "member", "lead_day"]
    ).reset_index(drop=True)


def write_cycle_csv(frame: pd.DataFrame, init: date, cycle_hour: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # "forecast" in the filename is a real signal to the schema mapper, but the live
    # ingest path passes explicit mappings rather than relying on it.
    path = out_dir / f"gefs_operational_forecast_{init:%Y%m%d}_{cycle_hour}z.csv"
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# cycle scheduling helpers
# --------------------------------------------------------------------------------------

def latest_expected_cycle(now: Optional[datetime] = None, lag_hours: float = 5.5,
                          cycles: Optional[list] = None) -> tuple:
    """Most recent (init_date, cycle_hour) that should have finished publishing."""
    now = now or datetime.now(timezone.utc)
    cycles = cycles or ["00", "06", "12", "18"]
    hours = sorted(int(c) for c in cycles)

    probe = now - timedelta(hours=lag_hours)
    day = probe.date()
    for _ in range(3):
        for h in reversed(hours):
            issued = datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc)
            if issued <= now - timedelta(hours=lag_hours):
                return day, f"{h:02d}"
        day = day - timedelta(days=1)
    raise CycleUnavailable("no cycle old enough to be published")


def recent_cycles(count: int, now: Optional[datetime] = None, lag_hours: float = 5.5,
                  cycles: Optional[list] = None, stride_hours: int = 24) -> list:
    """The `count` most recent cycles, walking back `stride_hours` at a time.

    Used by the backfill: a stride of 24 h spreads cycles across days rather than pulling
    four cycles of the same day, which carry nearly the same information.
    """
    init, hh = latest_expected_cycle(now, lag_hours, cycles)
    anchor = datetime(init.year, init.month, init.day, int(hh), tzinfo=timezone.utc)
    out = []
    for i in range(count):
        t = anchor - timedelta(hours=stride_hours * i)
        out.append((t.date(), f"{t.hour:02d}"))
    return out
