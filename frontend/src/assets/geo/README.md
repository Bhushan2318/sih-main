# Map geometry

Two files, both imported by `IndiaChoroplethMap.tsx`. Neither is covered by the
repository's MIT licence; each keeps its source's terms, recorded per file in
[`REUSE.toml`](../../../../REUSE.toml).

| File | What it is | Source and terms |
|---|---|---|
| `india_districts.topojson` | The 666 districts the map draws, simplified for the web by `backend/scripts/build_district_geo.py` from the same geometry the backend aggregates with | GADM 4.1, India admin-2 ([gadm.org](https://gadm.org)): free for academic and other non-commercial use; redistribution or commercial use needs GADM's prior permission. See [`LICENSES/LicenseRef-GADM.txt`](../../../../LICENSES/LicenseRef-GADM.txt) |
| `claimed_territory.geojson` | Areas India claims but does not administer (Gilgit-Baltistan, Aksai Chin, the Shaksgam Valley), which GADM has no district polygons for; built by `backend/scripts/build_claimed_territory_geo.py` | Natural Earth, public domain ([naturalearthdata.com](https://www.naturalearthdata.com)). See [`LICENSES/LicenseRef-NaturalEarth.txt`](../../../../LICENSES/LicenseRef-NaturalEarth.txt) |

Before changing either file, read
[`docs/boundary-geometry-licensing.md`](../../../../docs/boundary-geometry-licensing.md). It
covers India's 2021 geospatial guidelines, which name Survey of India data as the standard
for political maps of India.

The earlier state-level map (`india_states.topojson`, from `udit-001/india-maps-data`,
which publishes no licence) was removed on 2026-09-30, once nothing imported it.
