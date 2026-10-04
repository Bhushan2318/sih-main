"""Moisture and wind arithmetic shared by both sides of the bust label.

The forecast fetches (scripts/fetch_gefs_reforecast_sample.py, app/live/gefs.py) and the
observation fetches (scripts/fetch_era5_cds_district_observations.py,
app/live/observations.py) import these from here rather than
keeping a copy each. Two copies of a saturation formula drift, and the drift is invisible:
both keep producing plausible humidities, and the humidity bust label quietly records the
difference between them.

Every formula is Bolton (1980), the one the forecast side has always used.
"""
from __future__ import annotations

import numpy as np

# Ratio of the gas constants of dry air and water vapour, and its complement.
_EPS = 0.622
_ONE_MINUS_EPS = 0.378


def saturation_vapour_pressure_pa(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [Pa], Bolton (1980), temperature in kelvin."""
    t_k = np.asarray(t_k, dtype=float)
    return 611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))


def specific_humidity_from_dewpoint(td_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Specific humidity [kg/kg] from dewpoint [K] and pressure [Pa].

    The vapour pressure is es(Td); q = eps*e / (p - (1-eps)*e), the exact inverse of the
    e = q*p / (eps + (1-eps)*q) used by `rh_from_specific_humidity`.
    """
    e = saturation_vapour_pressure_pa(td_k)
    p_pa = np.asarray(p_pa, dtype=float)
    return _EPS * e / (p_pa - _ONE_MINUS_EPS * e)


def specific_humidity_from_rh(rh_pct: np.ndarray, t_k: np.ndarray,
                              p_pa: np.ndarray) -> np.ndarray:
    """Specific humidity [kg/kg] from relative humidity [%], temperature [K], pressure [Pa].

    The exact inverse of `rh_from_specific_humidity` (below 100%): e = RH * es(T), then
    q = eps*e / (p - (1-eps)*e). The live GEFS feed carries RH but not specific humidity at
    0.25 deg, so this recovers each cell's q before anything is averaged.
    """
    e = np.asarray(rh_pct, dtype=float) / 100.0 * saturation_vapour_pressure_pa(t_k)
    p_pa = np.asarray(p_pa, dtype=float)
    return _EPS * e / (p_pa - _ONE_MINUS_EPS * e)


def rh_from_specific_humidity(q: np.ndarray, t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Relative humidity [%] from specific humidity, temperature, pressure.

    Bolton (1980) saturation vapour pressure; standard q -> vapour-pressure inversion.
    """
    e = q * p_pa / (_EPS + _ONE_MINUS_EPS * q)
    es = saturation_vapour_pressure_pa(t_k)
    return np.clip(100.0 * e / es, 0.0, 100.0)


def wind_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Speed [m/s] and meteorological direction [deg, the direction the wind blows FROM]."""
    spd = np.sqrt(u**2 + v**2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return spd, direction
