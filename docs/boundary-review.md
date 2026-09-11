# Boundary depiction and licensing review

Research only, 2026-09-12. No geometry, code or asset was changed. `CLAUDE.md` says the
display depiction is under review separately from the geometry used for aggregation and
must not be changed without checking that review; this **is** that review's first pass.

**Not legal advice.** Two of the findings below turn on Indian law and on terms that are
not published in full. They need a person to confirm before anything is republished.

## What the project ships today

One script, `backend/scripts/build_district_geo.py`, derives everything from **GADM 4.1
India admin-2** (`https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip`),
with the three documented corrections: Ladakh reassigned out of Jammu and Kashmir (GADM
predates the 2019 reorganisation), the `Z01`/`Z04`/`Z05` disputed-territory features
dissolved into their parent districts, and Delhi relabelled.

It writes both:

| output | role | rendered? |
|---|---|---|
| `backend/data/geo/district_grid_weights.parquet` | the area-weight table both forecasts and observations are aggregated through | no, internal |
| `backend/data/geo/india_districts.json` / `.geojson` | the district registry | no, internal |
| `frontend/src/assets/geo/india_districts.topojson` (383,146 bytes, measured) | drawn in the browser, served publicly | **yes** |

So a single GADM-derived pipeline feeds both concerns. Separating them is the point.

## Finding 1 - the licensing problem is the concrete one

GADM's licence (gadm.org/license.html): *"The data are freely available for academic use
and other non-commercial use. Redistribution or commercial use is not allowed without
prior permission."*

Redistribution is prohibited **regardless** of being non-commercial. Sanket publishes a
383 KB TopoJSON derived from GADM on a live public site, which is redistribution of
derived geometry. That is a licence problem today, independent of whether the depiction
is politically correct, and it is the finding with the clearest remedy.

Using GADM to compute the internal weight table is *use*, not redistribution, and sits
inside "academic and other non-commercial use". Nothing here says the 0.9896-correlation
aggregation work has to be redone.

## Finding 2 - the legal regime is more permissive than it used to be, but not silent

India's **Guidelines for Acquiring and Producing Geospatial Data (DST, 15 Feb 2021)**
removed the old approval regime: *"there shall be no requirement for prior approval,
security clearance, license or any other restrictions on the collection, generation,
preparation, dissemination, storage, publication, updating and/or digitization of
Geospatial Data and Maps within the territory of India."* Compliance is self-certified.

The same guidelines keep a standard for what a political map should show: *"For political
Maps of India of any scale including national, state and other boundaries, SoI published
maps or SoI digital boundary data are the standard to be used, which shall be made easily
downloadable for free."*

Separately, secondary sources describe criminal exposure for publishing India's external
boundary incorrectly (one cites up to six months and/or a fine). That framing may pre-date
the 2021 liberalisation, and the 2021 guidelines are about producing geospatial data
rather than repealing boundary-depiction law. **Unverified; needs a lawyer.** The exposure,
if any, attaches to the *external* boundary and to Jammu and Kashmir / Ladakh, which is
exactly where GADM is weakest and where our corrections are applied by hand.

## Finding 3 - Survey of India's own data is not straightforwardly available to us

Survey of India distributes an administrative boundary database up to district level as
shapefiles, and its pricing policy lists those district-level products at **"0/-"**. But
the portal states the free data is *"free for users registered as 'Government Users' on
this portal"*, and publishes no terms for redistribution by anyone else. So the officially
sanctioned source is, in practice, gated. Confirming whether a student project can obtain
and republish it needs a human to ask them.

## Finding 4 - replacement candidates for the display layer

| source | licence | vintage | verdict |
|---|---|---|---|
| **data.gov.in "Admin Boundaries"** (State/District/Block, entire country) | **GODL-India** - worldwide, royalty-free, adapt and publish derivative works, commercial and non-commercial, attribution required | not confirmed | **Best licence fit.** The catalogue page confirms District + GODL, but the site is JavaScript-rendered and blocked automated fetching, so the actual file, its format and whether it is post-2019 are **unverified**. Needs a browser. |
| `datameet/maps` Districts | *"The dataset is shared under Creative Commons Attribution 2.5 India license."* - redistribution allowed with attribution | Census_2001 and Census_2011 only; external boundary "derived from the PC 2014 shapefile"; names from the Census Administrative Atlas | Usable, but pre-2019, so it needs the same Ladakh correction we already apply to GADM. No disputed-territory disclaimer. |
| `ramSeraph/indian_admin_boundaries` | repo licence field is `NOASSERTION`/"Other"; CC0 stated only in a release note | **historical**: decadal district boundaries 1941-2001 plus change logs to 2024, from indiastatestory.in | Not a current district layer. Earlier note calling this the best candidate was wrong. |

## Recommendation

1. **Keep GADM for aggregation.** Internal use, licence-compatible, already validated.
2. **Replace the published TopoJSON** with a source that permits redistribution - GODL
   (data.gov.in) first choice, datameet CC-BY 2.5 second - and re-apply the same three
   corrections plus attribution. This is a swap of one asset; no model, metric or weight
   table changes, so nothing about the scores moves.
3. **Do not treat the swap as closing the depiction question.** Whether the drawn boundary
   matches the official depiction is a separate check, and it is the half with real legal
   exposure.
4. **Attribution page (Workstream F6)** must then name the source, its licence and the
   required citation.

## Open, needs a human

- Confirm the data.gov.in district file: vintage, format, whether Ladakh is separate, and
  that GODL is the licence attached to that specific resource.
- Confirm current boundary-depiction law with someone qualified, specifically for a
  publicly hosted web map.
- Decide whether to ask Survey of India for access, or GADM for redistribution permission
  for this specific non-commercial project.
