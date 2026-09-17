# Boundary geometry: sourcing and licensing review (F1)

Written 2026-09-16. This is a research writeup, not a code change. It answers three
questions the team brief asked and flagged as open: is there a Survey of India (SoI)
requirement that applies here, what does GADM's licence actually say, and is Datameet a
viable alternative. All three are answered below from primary sources, cited with the
exact clause or line quoted — nothing here is paraphrased from memory.

## What the codebase actually uses today

Checked directly, not assumed:

- **Aggregation geometry** (`district_grid_weights.parquet`, the area-weighted mean of
  every 0.25° GEFS cell a district polygon overlaps) — GADM 4.1 India admin-2, per
  `backend/scripts/build_district_geo.py`'s own docstring: *"Source ... GADM 4.1 India
  level-2 (https://gadm.org)"*.
- **Display geometry** (the choropleth map users actually see) — **also** GADM 4.1, via
  the same script's `mapshaper` pipeline into
  `frontend/src/assets/geo/india_districts.topojson` (confirmed: 666 features, properties
  `region_id`/`region_name`/`state_id`/`state_name`, matching the aggregation file's
  region count exactly). `IndiaChoroplethMap.tsx` imports this file directly.
- **`frontend/src/assets/geo/india_states.topojson`** — a *different*, Datameet-sourced
  file (see its own `README.md` in that folder), vendored during the earlier state-level
  build. Grepped the frontend source: it is **no longer imported anywhere**. Dead weight,
  not the current display path.

**This means CLAUDE.md's note that "the boundary depiction used for display is under
review separately from the geometry used for aggregation" is now stale.** That was true
when the map was state-level (Datameet for display, GADM for aggregation, per
`docs/plan.md` line 45). Since the districts migration (`82af129`/`a24a663`), both layers
are the same GADM file. There is one geometry now, not two — which simplifies this
review, but also means whatever is true of GADM's licence and India's boundary rules
applies to what every visitor sees, not just to an internal computation.

## What GADM's licence says

Fetched `gadm.org/license.html` directly. Its own text:

- *"[the data] are freely available for academic use and other non-commercial use."*
- *"Redistribution or commercial use is not allowed without prior permission."*
- No explicit attribution clause in the licence text itself (attribution is still good
  practice and is what the "Data & attribution" section added for F6 does).

**For this project as it stands** — a non-commercial Smart India Hackathon submission —
GADM's terms are satisfied. They would **not** be satisfied without contacting GADM first
if this were ever deployed as a paid product, sold, or operated commercially, since that
crosses into the "commercial use" the licence reserves.

## What India's own rules say — this is the part that actually matters

The relevant instrument is the Department of Science & Technology's *"Guidelines for
acquiring and producing Geospatial Data and Geospatial Data Services including Maps"*
(DST F.No.SM/25/02/2020 (Part-I), dated 15 February 2021 — the 2021 liberalisation that
removed most prior-approval requirements). Fetched the official PDF from
`dst.gov.in/sites/default/files/Final Approved Guidelines on Geospatial Data.pdf` and
extracted its text directly. Two clauses apply here, quoted verbatim:

> **(xiii)** For political Maps of India of any scale including national, state and other
> boundaries, SoI published maps or SoI digital boundary data are the standard to be
> used, which shall be made easily downloadable for free and their digital display and
> printing shall be permissible. Others may publish such maps that adhere to these
> standards.

> **(xv)** Any violation of these guidelines will be dealt with under the applicable laws.

Clause (ii)(1) is the general liberalisation — no prior approval, licence, or security
clearance is required to collect, generate, publish, or display geospatial data covering
India. **Clause (xiii) is the one specific carve-out to that liberalisation**, and it
applies exactly to what this project renders: a political map of India, at district
scale, showing state and district boundaries. The standard it names is SoI's own
published boundary data — not "any boundary source that includes the right territories."

### Where this project already gets the intent right

`build_district_geo.py`'s own docstring documents a deliberate design choice that already
aligns with India's official position, independent of this review: GADM tags Jammu &
Kashmir, Ladakh, and ten Arunachal Pradesh districts with a `Z01`/`Z04`/`Z05`/`Z07`/`Z09`
prefix (its own signal that it treats the boundary there as internationally contested)
with **no separate `IND.*` feature for the same area**. The pipeline keeps every one of
those features regardless of prefix and dissolves duplicates into one region per
district, specifically so that "filtering to `IND.*` would erase Jammu and Kashmir,
Ladakh and much of Arunachal Pradesh from the map" — i.e., every district is drawn to the
full extent of India's claim, not GADM's disputed-territory framing. That is the right
call and it is already made.

### Where it is still open

Getting the *inclusion* right is not the same as the boundary *lines themselves* matching
SoI's published boundary data, which is what clause (xiii) actually names as the
standard. GADM's line-level tracing of contested boundaries (the Line of Control, the Line
of Actual Control, the Aksai Chin depiction, state-level lines within Arunachal Pradesh)
has no guarantee of matching SoI's official cartography pixel-for-pixel, because it is
compiled from a different source pipeline entirely. Nobody on this team has compared the
two line-by-line, and no docs anywhere in this repo (checked `docs/`, both locally and on
`origin/develop`) record that comparison having been done. **That comparison, not the
inclusion question, is what F1 leaves open.**

## Datameet as an alternative

`frontend/src/assets/geo/india_states.topojson`'s own `README.md` already documents it:
vendored from `udit-001/india-maps-data`, which "curates publicly available Survey of
India / Census 2011 administrative boundary data specifically for web choropleth use."
Fetched the upstream `datameet/maps` repository directly to check licensing: its README
states *"all datasets in this repository is shared under CC BY 4.0 license"* (with an MIT
licence on the repository's own code, not the data). Datameet's boundaries are
SoI/Census-2011-derived by design, which is a materially closer match to clause (xiii)'s
named standard than GADM is.

Two caveats found while checking this, both already flagged in that file's own README and
worth repeating here rather than glossing over:
- *"No explicit license is published for the geometry itself"* in the vendored copy's own
  disclaimer — the CC BY 4.0 statement is on the upstream `datameet/maps` repo, and it is
  not yet confirmed that the specific file vendored here (`udit-001/india-maps-data`,
  one hop removed from `datameet/maps`) carries the same terms forward.
- Datameet's provenance documentation, per that same fetch, "does not explicitly detail
  the original source or provenance of the boundary data itself" beyond crediting the
  curating community — one hop short of the primary SoI source itself.
- The vendored file is state-level plus an unused district object (726 features, a
  different count from GADM's 666 — GADM's Telangana-aware admin-2 split does not
  necessarily line up one-to-one with Datameet's district set), so swapping display
  geometry to it is not a drop-in file replacement; it would need its own review of
  region-ID mapping before use.

## Recommendation

1. **No code change needed for the hackathon submission itself.** GADM's non-commercial
   terms are satisfied by this project's current status, and the region-inclusion
   question (which the project actually controls) is already handled correctly.
2. **Line-level fidelity to SoI's official boundary data is the one genuinely open
   question**, and it is a display-geometry question only — the aggregation geometry's
   correctness does not depend on matching SoI's cartography, since GEFS grid cells and
   area weights are a scientific computation, not a published political map.
3. Before any deployment beyond the hackathon demo — anything a Government of India body
   would host, cite, or link to as a public tool — either (a) obtain and compare SoI's own
   digital boundary data against the current GADM file at the line level, or (b) replace
   the display layer with an SoI-sourced or SoI-verified alternative. Datameet is the best
   candidate found, but its own provenance chain needs one more hop of verification before
   it can be treated as equivalent to SoI's own data.
4. Delete the now-dead `frontend/src/assets/geo/india_states.topojson` and its `README.md`
   once a display-geometry decision is made, rather than leaving an unused file with its
   own (different) licensing story sitting in the bundle.

Sources fetched directly for this writeup:
- https://gadm.org/license.html
- https://github.com/datameet/maps
- https://dst.gov.in/sites/default/files/Final%20Approved%20Guidelines%20on%20Geospatial%20Data.pdf
