"""C3 - the MJO (Madden-Julian Oscillation) index, one row per calendar day.

Why NOAA PSL's OMI, not BOM's RMM
-----------------------------------
BOM's canonical RMM index
(http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt) is the index most
literature cites, but it returns HTTP 403 - "The Bureau of Meteorology website does not
support web scraping" - for any automated request. Verified 2026-09-18, not assumed; the
anonymous FTP mirror BOM's own error page suggests (ftp.bom.gov.au/anon/home/ncc/www/sco/)
does not carry it either. This repo does not circumvent an explicit anti-scraping notice.

NOAA PSL hosts the OMI (OLR-based MJO Index, Kiladis et al. 2014) at
https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt: real, no-auth, government-hosted, updated
daily since 1991. PSL's own page (https://psl.noaa.gov/mjo/mjoindex/, fetched 2026-09-18)
documents the transform to the standard RMM convention: "the sign of OMI PC1 and the PC
ordering should be reversed, so that OMI(PC2) is analogous to RMM(PC1) and -OMI(PC1) is
analogous to RMM (PC2)" - i.e. RMM1 = OMI(PC2), RMM2 = -OMI(PC1) - and states the
correlation between OMI and RMM exceeds 0.93 for PC1, PC2 and amplitude.

Not built: MISO (the Indian-region monsoon analogue C3 was also named for) and a discrete
1-8 MJO phase. Neither has a source/convention this repo could independently verify -
see docs/known-issues.md.

File format (whitespace-separated, no header): year month day PC1 PC2 amplitude. The
amplitude column is the file's own sqrt(PC1^2+PC2^2), rotation-invariant, so it is used
directly rather than recomputed from the transformed values.

Re-fetched each run, not idempotent like the geo builds: this is a growing daily time
series (~13,000 rows, well under 1 MB), not a multi-GB archive, so a full re-fetch is the
simplest correct way to pick up new days. Not a fetch this project needs to fetch once
and never touch again like the 4.2 TB GEFS archive.

    python -m scripts.fetch_mjo_index
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from app.config import settings
from app.db.base import resolve_path

URL = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
OUT_FILENAME = "mjo_omi_index.parquet"
SOURCE = ("NOAA PSL OMI index (https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt), "
         "transformed to the RMM1/RMM2 convention per PSL's documented mapping")


def _omi_to_rmm(pc1: float, pc2: float) -> tuple[float, float]:
    """RMM1 = OMI(PC2), RMM2 = -OMI(PC1) - PSL's documented convention, not this
    project's own derivation."""
    return pc2, -pc1


def parse(text: str) -> pd.DataFrame:
    raw = pd.read_csv(
        io.StringIO(text), sep=r"\s+", header=None,
        names=["year", "month", "day", "pc1", "pc2", "amplitude"])
    rmm1, rmm2 = _omi_to_rmm(raw["pc1"].to_numpy(), raw["pc2"].to_numpy())
    return pd.DataFrame({
        "date": pd.to_datetime(raw[["year", "month", "day"]]),
        "mjo_rmm1": rmm1,
        "mjo_rmm2": rmm2,
        "mjo_amplitude": raw["amplitude"].to_numpy(),
        "source": SOURCE,
    })


def fetch() -> pd.DataFrame:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    out = parse(text)
    if out.empty:
        raise RuntimeError(f"{URL} returned no rows - refusing to write an empty index")
    return out


def main() -> None:
    out = fetch()
    path = resolve_path(settings.data_dir) / OUT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"{len(out)} days ({out['date'].min().date()}..{out['date'].max().date()}) "
        f"-> {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
