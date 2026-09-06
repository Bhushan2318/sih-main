"""Storage format for gridded forecast fields.

The tabular pipeline reduces each forecast to one number per district, which is all the
XGBoost models need and throws away the spatial pattern that produced it. The
convolutional model needs that pattern, so the fetch keeps a second artifact: the raw
0.25 deg fields over India and its surrounding seas.

Two decisions make that affordable. Stated here because they are the difference between
2 GB and 11 GB over the archive, and because both are visible in the model's inputs.

*Ensemble mean and spread, not five raw members.* Five members at float32 is 11.25 GB
over 344 cycles; mean and spread at int16 is 2.25 GB. This is not only a size argument -
mean and spread are the standard predictors for neural post-processing of an ensemble
(Rasp and Lerch, 2018), because they are what the ensemble is actually saying: its best
guess and its own confidence in it. Per-member detail beyond that is noise at 5 members.

*Scaled int16, not float32.* Each plane is stored as integer multiples of a physical
quantum - 0.01 in the variable's own unit - offset by its mean. INT_NAN is reserved as
the missing-value sentinel so a gap stays a gap: no field is ever silently filled with a
zero or an interpolated value.

The quantum matters for size as much as for precision. Stretching each plane across the
full int16 range, the obvious encoding, maximises entropy and left the archive
incompressible: 6.43 MB per cycle against 6.54 MB raw, a 2% saving. Quantising to a
physical step instead keeps neighbouring cells close in integer value, and splitting the
high and low bytes into separate planes then gives zlib a high-byte plane that is almost
flat. Together: 3.98 MB per cycle, 1.37 GB over 344 cycles rather than 2.25 GB - which is
also what brings it under GitHub's 2 GB per-asset limit. A plane whose range is too wide
for the quantum widens its own step rather than clipping, so an extreme rainfall day is
stored coarser but never wrong.

The domain deliberately extends past the coastline (lon 65-100, lat 2-38). A network that
only sees Indian land cannot see the Arabian Sea and Bay of Bengal depressions or the
westerly troughs that cause the busts it is asked to predict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_DEG = 0.25

# lon/lat bounds of the stored domain, inclusive.
DOMAIN_LON = (65.0, 100.0)
DOMAIN_LAT = (2.0, 38.0)

STATS = ("mean", "spread")

INT_NAN = np.int16(-32768)   # reserved; the value range is [-32767, 32767]
_INT_MAX = 32767.0

# Target precision, in each variable's own physical unit. Every stored variable - degrees
# Celsius, millimetres, hectopascals, per cent, m/s - is comfortably resolved by 0.01,
# which is one to two orders finer than any of them is forecast to.
QUANTUM = 0.01


def domain_coords() -> tuple[np.ndarray, np.ndarray]:
    """(lats, lons) of the stored domain. Latitudes ascend, longitudes are -180..180."""
    lats = np.arange(DOMAIN_LAT[0], DOMAIN_LAT[1] + GRID_DEG / 2, GRID_DEG)
    lons = np.arange(DOMAIN_LON[0], DOMAIN_LON[1] + GRID_DEG / 2, GRID_DEG)
    return lats.astype(np.float32), lons.astype(np.float32)


@dataclass(frozen=True)
class GridBundle:
    """One initialisation's fields: values[var, lead, stat, lat, lon] in physical units."""

    init_date: str
    variables: tuple[str, ...]
    leads: tuple[int, ...]
    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray          # float32, NaN where missing

    def __post_init__(self):
        want = (len(self.variables), len(self.leads), len(STATS),
                len(self.lats), len(self.lons))
        if self.values.shape != want:
            raise ValueError(f"values shape {self.values.shape} != {want}")

    def field(self, variable: str, lead: int, stat: str = "mean") -> np.ndarray:
        return self.values[self.variables.index(variable),
                           self.leads.index(lead), STATS.index(stat)]

    def flat_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat (lat, lon) arrays matching a ravelled field - what the district
        aggregator expects."""
        la, lo = np.meshgrid(self.lats, self.lons, indexing="ij")
        return la.ravel(), lo.ravel()


def _encode_plane(plane: np.ndarray) -> tuple[np.ndarray, float, float]:
    """float -> int16 in steps of QUANTUM about the plane's midpoint.

    Returns (ints, scale, offset). The step widens only if the plane's own range will not
    fit in int16 at QUANTUM - a 900 mm rainfall day, say - so precision degrades on the
    rare wide plane instead of the values being clipped to the ends of the range.
    """
    finite = np.isfinite(plane)
    if not finite.any():
        return np.full(plane.shape, INT_NAN, dtype=np.int16), QUANTUM, 0.0
    lo = float(plane[finite].min())
    hi = float(plane[finite].max())
    span = hi - lo
    scale = max(QUANTUM, span / (2 * _INT_MAX))
    offset = lo + span / 2.0
    out = np.full(plane.shape, INT_NAN, dtype=np.int16)
    scaled = np.rint((plane[finite] - offset) / scale)
    out[finite] = np.clip(scaled, -_INT_MAX, _INT_MAX).astype(np.int16)
    return out, scale, offset


def _split_bytes(ints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int16 -> (high byte, low byte) planes.

    Stored separately so each is compressed on its own. Neighbouring cells differ by only
    a few quanta, so the high-byte plane is nearly constant and zlib collapses it; mixed
    together as native int16 the two bytes interleave and neither compresses.
    """
    u = (ints.astype(np.int32) + 32768).astype(np.uint16)
    return (u >> 8).astype(np.uint8), (u & 0xFF).astype(np.uint8)


def _join_bytes(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    u = (hi.astype(np.uint16) << 8) | lo.astype(np.uint16)
    return (u.astype(np.int32) - 32768).astype(np.int16)


def save_bundle(path: Path, bundle: GridBundle) -> Path:
    """Write one initialisation as a compressed npz."""
    n_var, n_lead, n_stat = len(bundle.variables), len(bundle.leads), len(STATS)
    ints = np.empty(bundle.values.shape, dtype=np.int16)
    scales = np.ones((n_var, n_lead, n_stat), dtype=np.float64)
    offsets = np.zeros((n_var, n_lead, n_stat), dtype=np.float64)
    for v in range(n_var):
        for l in range(n_lead):
            for s in range(n_stat):
                ints[v, l, s], scales[v, l, s], offsets[v, l, s] = \
                    _encode_plane(bundle.values[v, l, s])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hi, lo = _split_bytes(ints)
    np.savez_compressed(
        path,
        data_hi=hi, data_lo=lo, scale=scales, offset=offsets,
        lats=bundle.lats, lons=bundle.lons,
        variables=np.array(bundle.variables), leads=np.array(bundle.leads),
        stats=np.array(STATS), init_date=np.array(bundle.init_date),
    )
    return path


def load_bundle(path: Path) -> GridBundle:
    with np.load(Path(path), allow_pickle=False) as z:
        ints = _join_bytes(z["data_hi"], z["data_lo"])
        vals = np.where(ints == INT_NAN, np.nan,
                        ints.astype(np.float64) * z["scale"][..., None, None]
                        + z["offset"][..., None, None]).astype(np.float32)
        return GridBundle(
            init_date=str(z["init_date"]),
            variables=tuple(str(v) for v in z["variables"]),
            leads=tuple(int(x) for x in z["leads"]),
            lats=z["lats"], lons=z["lons"], values=vals,
        )


def subset_to_domain(lats: np.ndarray, lons: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Regrid a decoded GRIB field onto the stored domain.

    GEFS publishes latitude descending and longitude on 0..360; the stored domain ascends
    and uses -180..180. Cells the source does not cover become NaN rather than being
    filled by the nearest neighbour - an invented value at the domain edge would be
    indistinguishable from a real one to the network.
    """
    want_lats, want_lons = domain_coords()
    src_lat = {int(round(float(v) / GRID_DEG)): i for i, v in enumerate(lats)}
    src_lon = {int(round((float(v) % 360.0) / GRID_DEG)): i for i, v in enumerate(lons)}

    out = np.full((len(want_lats), len(want_lons)), np.nan, dtype=np.float32)
    for i, la in enumerate(want_lats):
        si = src_lat.get(int(round(float(la) / GRID_DEG)))
        if si is None:
            continue
        for j, lo in enumerate(want_lons):
            sj = src_lon.get(int(round((float(lo) % 360.0) / GRID_DEG)))
            if sj is not None:
                out[i, j] = field[si, sj]
    return out
