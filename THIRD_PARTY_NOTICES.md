# Third-party notices

Sanket's own code is released under the [MIT Licence](LICENSE). This file lists everything
else the project is built on: the data, the Python and JavaScript libraries, the fonts and
the container images. For each it gives the licence and any attribution the source asks for.

- [`REUSE.toml`](REUSE.toml) records the copyright and licence of every file in this
  repository, and `reuse lint` checks it ([REUSE 3.3](https://reuse.software)).
- [`LICENSES/`](LICENSES/) holds the full text of every licence that applies to a file here.
- [`THIRD_PARTY_LICENSES.txt`](THIRD_PARTY_LICENSES.txt) holds the full licence text of every
  package shipped to users: the 49 Python packages in the serving image and the
  52 JavaScript packages bundled into the site. The site serves the JavaScript part
  itself at [`/third-party-licenses.txt`](https://sanket-a0dd.onrender.com/third-party-licenses.txt).

Generated 2026-09-30. The Python sets were resolved with `uv pip compile` for Linux x86_64 and
Python 3.12, the platform of the Docker image. The licence texts were read from those exact
wheels. The JavaScript set comes from `frontend/package-lock.json`.

## Summary

| What | Count | Licences |
|---|---|---|
| Data sources | 10 | see [Data](#data) |
| Python packages in the serving image | 49 | MIT (21), BSD-3-Clause (14), Apache-2.0 (5), MPL-2.0 (1), MIT AND PSF-2.0 (1), BSD-2-Clause AND Apache-2.0 WITH LLVM-exception (1), BSD-2-Clause (1), Apache-2.0 OR BSD-2-Clause (1), Apache-2.0 AND BSD-3-Clause (1), MPL-2.0 AND MIT (1), PSF-2.0 (1), MIT OR Apache-2.0 (1) |
| Python packages added for tests | 8 | see [Python packages](#python-packages) |
| Python packages added for live ingestion | 23 | see [Python packages](#python-packages) |
| Python packages added for CNN training | 26 | see [Python packages](#python-packages) |
| JavaScript packages bundled into the site | 52 | MIT (35), ISC (12), BSD-3-Clause (4), MIT AND ISC (1) |
| JavaScript build and test tools | 343 | see [JavaScript packages](#javascript-packages) |
| Web fonts | 3 | OFL-1.1 |

## Data

Every value Sanket shows comes from one of these. The terms quoted in `LICENSES/` were
fetched from each provider on 2026-09-30.

| Source | Used for | In this repository | Terms | Attribution the source asks for |
|---|---|---|---|---|
| NOAA GEFSv12 reforecast (AWS Open Data, `noaa-gefs-retrospective`) | Training forecasts, 2000–2019 | `backend/data/samples/gefs_reforecast_india_2019.parquet` | NOAA open data: "open to the public and can be used as desired" ([LicenseRef-NOAA-Open-Data](LICENSES/LicenseRef-NOAA-Open-Data.txt)) | NOAA requests attribution; do not imply NOAA endorsement, or present modified data as original NOAA data |
| NOAA GEFS operational feed (NOMADS, AWS) | Live forecasts | none (fetched at run time) | As above | As above |
| ERA5 reanalysis, Copernicus Climate Change Service (Climate Data Store) | Observations for training and verification | none (fetched at run time) | [CC-BY-4.0](LICENSES/CC-BY-4.0.txt) | "Generated using Copernicus Climate Change Service information [Year]", or "Contains modified Copernicus Climate Change Service information [Year]" for derived data |
| ERA5 through the Open-Meteo Historical Weather API | Observations for recent days; the 2019 sample | `backend/data/samples/era5_observations_india_2019.parquet` | [CC-BY-4.0](LICENSES/CC-BY-4.0.txt) (Open-Meteo: "The data obtained through the API is provided under the terms of the CC-BY 4.0 licence") | Open-Meteo: "You must include a link next to any location Open-Meteo data are displayed", e.g. "Weather data by Open-Meteo.com" linking to https://open-meteo.com/. Also credit Copernicus, as above |
| GADM 4.1, India admin-2 | District boundaries, the weight table and the map | `backend/data/geo/india_districts.*`, `district_grid_weights.parquet`, `district_descriptors.parquet`, `frontend/src/assets/geo/india_districts.topojson` | [LicenseRef-GADM](LICENSES/LicenseRef-GADM.txt): free for academic and other non-commercial use; "Redistribution or commercial use is not allowed without prior permission" | None required; credited throughout |
| Natural Earth, disputed areas | The claimed-territory overlay | `frontend/src/assets/geo/claimed_territory.geojson` | [LicenseRef-NaturalEarth](LICENSES/LicenseRef-NaturalEarth.txt): public domain | None required ("Made with Natural Earth" if cited) |
| CGIAR-CSI hole-filled SRTM v4, through the Open-Elevation API | District elevation | `backend/data/geo/grid_elevation_m.parquet`, `district_descriptors.parquet` | [LicenseRef-CIAT-SRTM](LICENSES/LicenseRef-CIAT-SRTM.txt): no commercial use, resale or redistribution without CIAT's written permission | Acknowledge CIAT; cite Jarvis et al. (2008), Hole-filled seamless SRTM data V4 |
| NOAA PSL OLR MJO Index (OMI) | MJO features | `backend/data/mjo_omi_index.parquet` | [LicenseRef-NOAA-Open-Data](LICENSES/LicenseRef-NOAA-Open-Data.txt) | Cite Kiladis et al. (2014), Monthly Weather Review 142, 1697–1715 |
| MERRA-2 (NASA GMAO), through the NASA POWER API | Measuring the ground truth's own uncertainty | `docs/analysis/*.csv` | [LicenseRef-NASA-POWER](LICENSES/LicenseRef-NASA-POWER.txt) | Cite POWER's reference: "The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division." |
| IMD gauge-based gridded rainfall (India Meteorological Department, Pune) | Rainfall in some held-out years | none (fetched at run time) | IMD's own terms ([imdpune.gov.in](https://imdpune.gov.in/cmpg/Griddata/rainfall.php)) | Credit IMD |

Two sets of terms restrict redistribution, and both apply to files this public repository
contains. GADM does not allow redistribution without prior permission, and CIAT does not
allow it without written permission. The files are here for a non-commercial hackathon
prototype (see [`docs/boundary-geometry-licensing.md`](docs/boundary-geometry-licensing.md)).
Any wider use needs those permissions first.

## Fonts

Loaded from Google Fonts when the site opens; not stored in this repository.

| Font | Designer | Licence |
|---|---|---|
| [Archivo](https://fonts.google.com/specimen/Archivo) | Omnibus-Type | [OFL-1.1](https://github.com/google/fonts/blob/main/ofl/archivo/OFL.txt) |
| [Public Sans](https://fonts.google.com/specimen/Public+Sans) | USWDS, Dan Williams, Pablo Impallari, Rodrigo Fuenzalida | [OFL-1.1](https://github.com/google/fonts/blob/main/ofl/publicsans/OFL.txt) |
| [DM Mono](https://fonts.google.com/specimen/DM+Mono) | Colophon Foundry | [OFL-1.1](https://github.com/google/fonts/blob/main/ofl/dmmono/OFL.txt) |

## Container images

The `Dockerfile` builds on two official images, each carrying its own licences:
`node:20-alpine` for the build stage only (Node.js under the
[MIT licence](https://github.com/nodejs/node/blob/main/LICENSE), on Alpine Linux packages
under their own licences), and `python:3.12-slim` for the served image (Python under the
[PSF-2.0 licence](https://docs.python.org/3/license.html), on Debian packages under their own
licences).

## Python packages

### In the serving image (`backend/requirements.txt`)

| Package | Version | Licence |
|---|---|---|
| [annotated-types](https://pypi.org/project/annotated-types/0.8.0/) | 0.8.0 | MIT |
| [anyio](https://pypi.org/project/anyio/4.15.1/) | 4.15.1 | MIT |
| [certifi](https://pypi.org/project/certifi/2026.7.22/) | 2026.7.22 | MPL-2.0 |
| [charset-normalizer](https://pypi.org/project/charset-normalizer/3.5.2/) | 3.5.2 | MIT |
| [click](https://pypi.org/project/click/8.5.0/) | 8.5.0 | BSD-3-Clause |
| [cloudpickle](https://pypi.org/project/cloudpickle/3.1.2/) | 3.1.2 | BSD-3-Clause |
| [et-xmlfile](https://pypi.org/project/et-xmlfile/2.0.0/) | 2.0.0 | MIT |
| [fastapi](https://pypi.org/project/fastapi/0.115.14/) | 0.115.14 | MIT |
| [greenlet](https://pypi.org/project/greenlet/3.5.6/) | 3.5.6 | MIT AND PSF-2.0 |
| [h11](https://pypi.org/project/h11/0.16.0/) | 0.16.0 | MIT |
| [httptools](https://pypi.org/project/httptools/0.8.0/) | 0.8.0 | MIT |
| [idna](https://pypi.org/project/idna/3.20/) | 3.20 | BSD-3-Clause |
| [joblib](https://pypi.org/project/joblib/1.6.0/) | 1.6.0 | BSD-3-Clause |
| [llvmlite](https://pypi.org/project/llvmlite/0.49.0/) | 0.49.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception |
| [numba](https://pypi.org/project/numba/0.67.0/) | 0.67.0 | BSD-2-Clause |
| [numpy](https://pypi.org/project/numpy/1.26.4/) | 1.26.4 | BSD-3-Clause |
| [openpyxl](https://pypi.org/project/openpyxl/3.1.5/) | 3.1.5 | MIT |
| [packaging](https://pypi.org/project/packaging/26.3/) | 26.3 | Apache-2.0 OR BSD-2-Clause |
| [pandas](https://pypi.org/project/pandas/2.1.4/) | 2.1.4 | BSD-3-Clause |
| [pyarrow](https://pypi.org/project/pyarrow/16.1.0/) | 16.1.0 | Apache-2.0 |
| [pydantic](https://pypi.org/project/pydantic/2.8.2/) | 2.8.2 | MIT |
| [pydantic-core](https://pypi.org/project/pydantic-core/2.20.1/) | 2.20.1 | MIT |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/2.3.4/) | 2.3.4 | MIT |
| [pyshp](https://pypi.org/project/pyshp/3.1.6/) | 3.1.6 | MIT |
| [python-dateutil](https://pypi.org/project/python-dateutil/2.9.0.post0/) | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause |
| [python-dotenv](https://pypi.org/project/python-dotenv/1.2.3/) | 1.2.3 | BSD-3-Clause |
| [python-multipart](https://pypi.org/project/python-multipart/0.0.32/) | 0.0.32 | Apache-2.0 |
| [pytz](https://pypi.org/project/pytz/2026.4/) | 2026.4 | MIT |
| [pyyaml](https://pypi.org/project/pyyaml/6.0.3/) | 6.0.3 | MIT |
| [rapidfuzz](https://pypi.org/project/rapidfuzz/3.14.6/) | 3.14.6 | MIT |
| [requests](https://pypi.org/project/requests/2.34.2/) | 2.34.2 | Apache-2.0 |
| [scikit-learn](https://pypi.org/project/scikit-learn/1.4.2/) | 1.4.2 | BSD-3-Clause |
| [scipy](https://pypi.org/project/scipy/1.17.1/) | 1.17.1 | BSD-3-Clause |
| [shap](https://pypi.org/project/shap/0.45.1/) | 0.45.1 | MIT |
| [shapely](https://pypi.org/project/shapely/2.0.7/) | 2.0.7 | BSD-3-Clause |
| [six](https://pypi.org/project/six/1.17.0/) | 1.17.0 | MIT |
| [slicer](https://pypi.org/project/slicer/0.0.8/) | 0.0.8 | MIT |
| [sqlalchemy](https://pypi.org/project/sqlalchemy/2.0.54/) | 2.0.54 | MIT |
| [starlette](https://pypi.org/project/starlette/0.46.2/) | 0.46.2 | BSD-3-Clause |
| [threadpoolctl](https://pypi.org/project/threadpoolctl/3.7.0/) | 3.7.0 | BSD-3-Clause |
| [tqdm](https://pypi.org/project/tqdm/4.70.1/) | 4.70.1 | MPL-2.0 AND MIT |
| [typing-extensions](https://pypi.org/project/typing-extensions/4.16.0/) | 4.16.0 | PSF-2.0 |
| [tzdata](https://pypi.org/project/tzdata/2026.4/) | 2026.4 | Apache-2.0 |
| [urllib3](https://pypi.org/project/urllib3/2.8.0/) | 2.8.0 | MIT |
| [uvicorn](https://pypi.org/project/uvicorn/0.30.6/) | 0.30.6 | BSD-3-Clause |
| [uvloop](https://pypi.org/project/uvloop/0.22.1/) | 0.22.1 | MIT OR Apache-2.0 |
| [watchfiles](https://pypi.org/project/watchfiles/1.3.0/) | 1.3.0 | MIT |
| [websockets](https://pypi.org/project/websockets/12.0/) | 12.0 | BSD-3-Clause |
| [xgboost](https://pypi.org/project/xgboost/2.0.3/) | 2.0.3 | Apache-2.0 |

### Added for tests (`requirements-dev.txt`)

| Package | Version | Licence |
|---|---|---|
| [httpcore](https://pypi.org/project/httpcore/1.0.9/) | 1.0.9 | BSD-3-Clause |
| [httpx](https://pypi.org/project/httpx/0.27.2/) | 0.27.2 | BSD-3-Clause |
| [iniconfig](https://pypi.org/project/iniconfig/2.3.0/) | 2.3.0 | MIT |
| [pluggy](https://pypi.org/project/pluggy/1.6.0/) | 1.6.0 | MIT |
| [psutil](https://pypi.org/project/psutil/7.2.2/) | 7.2.2 | BSD-3-Clause |
| [pygments](https://pypi.org/project/pygments/2.21.0/) | 2.21.0 | BSD-2-Clause |
| [pytest](https://pypi.org/project/pytest/8.4.2/) | 8.4.2 | MIT |
| [sniffio](https://pypi.org/project/sniffio/1.3.1/) | 1.3.1 | MIT OR Apache-2.0 |

### Added for live ingestion (`requirements-live.txt`)

| Package | Version | Licence |
|---|---|---|
| [attrs](https://pypi.org/project/attrs/26.1.0/) | 26.1.0 | MIT |
| [cdsapi](https://pypi.org/project/cdsapi/0.7.7/) | 0.7.7 | Apache-2.0 |
| [cffi](https://pypi.org/project/cffi/2.1.1/) | 2.1.1 | MIT-0 |
| [cfgrib](https://pypi.org/project/cfgrib/0.9.15.1/) | 0.9.15.1 | Apache-2.0 |
| [cftime](https://pypi.org/project/cftime/1.6.6/) | 1.6.6 | MIT |
| [contourpy](https://pypi.org/project/contourpy/1.3.3/) | 1.3.3 | BSD-3-Clause |
| [cycler](https://pypi.org/project/cycler/0.12.1/) | 0.12.1 | BSD-3-Clause |
| [eccodes](https://pypi.org/project/eccodes/2.48.0/) | 2.48.0 | Apache-2.0 |
| [eccodeslib](https://pypi.org/project/eccodeslib/2.49.0.30/) | 2.49.0.30 | Apache-2.0 |
| [eckitlib](https://pypi.org/project/eckitlib/2.3.0.30/) | 2.3.0.30 | Apache-2.0 |
| [ecmwf-datastores-client](https://pypi.org/project/ecmwf-datastores-client/0.5.3/) | 0.5.3 | Apache-2.0 |
| [ecmwflibs](https://pypi.org/project/ecmwflibs/0.7.1/) | 0.7.1 | Apache-2.0 |
| [findlibs](https://pypi.org/project/findlibs/0.1.3/) | 0.1.3 | Apache-2.0 |
| [fonttools](https://pypi.org/project/fonttools/4.66.1/) | 4.66.1 | MIT |
| [imdlib](https://pypi.org/project/imdlib/0.1.21/) | 0.1.21 | MIT |
| [kiwisolver](https://pypi.org/project/kiwisolver/1.5.1/) | 1.5.1 | BSD-3-Clause |
| [matplotlib](https://pypi.org/project/matplotlib/3.11.2/) | 3.11.2 | LicenseRef-Matplotlib (PSF-based) |
| [multiurl](https://pypi.org/project/multiurl/0.3.9/) | 0.3.9 | Apache-2.0 |
| [netcdf4](https://pypi.org/project/netcdf4/1.7.4/) | 1.7.4 | MIT |
| [pillow](https://pypi.org/project/pillow/12.3.0/) | 12.3.0 | MIT-CMU |
| [pycparser](https://pypi.org/project/pycparser/3.0/) | 3.0 | BSD-3-Clause |
| [pyparsing](https://pypi.org/project/pyparsing/3.3.3/) | 3.3.3 | MIT |
| [xarray](https://pypi.org/project/xarray/2025.6.1/) | 2025.6.1 | Apache-2.0 |

### Added for CNN training (`requirements-train.txt`)

On Linux x86_64, PyTorch from PyPI also installs NVIDIA's CUDA libraries (the `nvidia-*`
packages below), which are proprietary software under
[NVIDIA's licence](https://docs.nvidia.com/cuda/eula/index.html). They are needed only to
train the convolutional challenger on a GPU, and never enter the serving image.

| Package | Version | Licence |
|---|---|---|
| [coloredlogs](https://pypi.org/project/coloredlogs/15.0.1/) | 15.0.1 | MIT |
| [filelock](https://pypi.org/project/filelock/4.0.7/) | 4.0.7 | MIT |
| [flatbuffers](https://pypi.org/project/flatbuffers/25.12.19/) | 25.12.19 | Apache-2.0 |
| [fsspec](https://pypi.org/project/fsspec/2026.9.0/) | 2026.9.0 | BSD-3-Clause |
| [humanfriendly](https://pypi.org/project/humanfriendly/10.0/) | 10.0 | MIT |
| [jinja2](https://pypi.org/project/jinja2/3.1.6/) | 3.1.6 | BSD-3-Clause |
| [markupsafe](https://pypi.org/project/markupsafe/3.0.3/) | 3.0.3 | BSD-3-Clause |
| [mpmath](https://pypi.org/project/mpmath/1.3.0/) | 1.3.0 | BSD-3-Clause |
| [networkx](https://pypi.org/project/networkx/3.7/) | 3.7 | BSD-3-Clause |
| [nvidia-cublas-cu12](https://pypi.org/project/nvidia-cublas-cu12/12.1.3.1/) | 12.1.3.1 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cuda-cupti-cu12](https://pypi.org/project/nvidia-cuda-cupti-cu12/12.1.105/) | 12.1.105 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cuda-nvrtc-cu12](https://pypi.org/project/nvidia-cuda-nvrtc-cu12/12.1.105/) | 12.1.105 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cuda-runtime-cu12](https://pypi.org/project/nvidia-cuda-runtime-cu12/12.1.105/) | 12.1.105 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cudnn-cu12](https://pypi.org/project/nvidia-cudnn-cu12/8.9.2.26/) | 8.9.2.26 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cufft-cu12](https://pypi.org/project/nvidia-cufft-cu12/11.0.2.54/) | 11.0.2.54 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-curand-cu12](https://pypi.org/project/nvidia-curand-cu12/10.3.2.106/) | 10.3.2.106 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cusolver-cu12](https://pypi.org/project/nvidia-cusolver-cu12/11.4.5.107/) | 11.4.5.107 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-cusparse-cu12](https://pypi.org/project/nvidia-cusparse-cu12/12.1.0.106/) | 12.1.0.106 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-nccl-cu12](https://pypi.org/project/nvidia-nccl-cu12/2.19.3/) | 2.19.3 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-nvjitlink-cu12](https://pypi.org/project/nvidia-nvjitlink-cu12/12.9.86/) | 12.9.86 | LicenseRef-NVIDIA-Proprietary |
| [nvidia-nvtx-cu12](https://pypi.org/project/nvidia-nvtx-cu12/12.1.105/) | 12.1.105 | LicenseRef-NVIDIA-Proprietary |
| [onnx](https://pypi.org/project/onnx/1.16.2/) | 1.16.2 | Apache-2.0 |
| [onnxruntime](https://pypi.org/project/onnxruntime/1.18.1/) | 1.18.1 | MIT |
| [protobuf](https://pypi.org/project/protobuf/7.36.2/) | 7.36.2 | BSD-3-Clause |
| [sympy](https://pypi.org/project/sympy/1.14.0/) | 1.14.0 | BSD-3-Clause |
| [torch](https://pypi.org/project/torch/2.2.2/) | 2.2.2 | BSD-3-Clause |

## JavaScript packages

### Bundled into the site

| Package | Version | Licence |
|---|---|---|
| [@babel/runtime](https://www.npmjs.com/package/@babel/runtime/v/7.29.7) | 7.29.7 | MIT |
| [@tanstack/query-core](https://www.npmjs.com/package/@tanstack/query-core/v/5.102.7) | 5.102.7 | MIT |
| [@tanstack/react-query](https://www.npmjs.com/package/@tanstack/react-query/v/5.102.7) | 5.102.7 | MIT |
| [@types/d3-array](https://www.npmjs.com/package/@types/d3-array/v/3.2.2) | 3.2.2 | MIT |
| [@types/d3-color](https://www.npmjs.com/package/@types/d3-color/v/2.0.6) | 2.0.6 | MIT |
| [@types/d3-ease](https://www.npmjs.com/package/@types/d3-ease/v/3.0.2) | 3.0.2 | MIT |
| [@types/d3-interpolate](https://www.npmjs.com/package/@types/d3-interpolate/v/3.0.4) | 3.0.4 | MIT |
| [@types/d3-path](https://www.npmjs.com/package/@types/d3-path/v/3.1.1) | 3.1.1 | MIT |
| [@types/d3-scale](https://www.npmjs.com/package/@types/d3-scale/v/4.0.9) | 4.0.9 | MIT |
| [@types/d3-shape](https://www.npmjs.com/package/@types/d3-shape/v/3.2.0) | 3.2.0 | MIT |
| [@types/d3-time](https://www.npmjs.com/package/@types/d3-time/v/3.0.4) | 3.0.4 | MIT |
| [@types/d3-timer](https://www.npmjs.com/package/@types/d3-timer/v/3.0.2) | 3.0.2 | MIT |
| [clsx](https://www.npmjs.com/package/clsx/v/2.1.1) | 2.1.1 | MIT |
| [commander](https://www.npmjs.com/package/commander/v/2.20.3) | 2.20.3 | MIT |
| [csstype](https://www.npmjs.com/package/csstype/v/3.2.3) | 3.2.3 | MIT |
| [d3-array](https://www.npmjs.com/package/d3-array/v/3.2.4) | 3.2.4 | ISC |
| [d3-color](https://www.npmjs.com/package/d3-color/v/2.0.0) | 2.0.0 | BSD-3-Clause |
| [d3-ease](https://www.npmjs.com/package/d3-ease/v/3.0.1) | 3.0.1 | BSD-3-Clause |
| [d3-format](https://www.npmjs.com/package/d3-format/v/3.1.2) | 3.1.2 | ISC |
| [d3-geo](https://www.npmjs.com/package/d3-geo/v/3.1.1) | 3.1.1 | ISC |
| [d3-interpolate](https://www.npmjs.com/package/d3-interpolate/v/2.0.1) | 2.0.1 | BSD-3-Clause |
| [d3-interpolate](https://www.npmjs.com/package/d3-interpolate/v/3.0.1) | 3.0.1 | ISC |
| [d3-path](https://www.npmjs.com/package/d3-path/v/3.1.0) | 3.1.0 | ISC |
| [d3-scale](https://www.npmjs.com/package/d3-scale/v/4.0.2) | 4.0.2 | ISC |
| [d3-shape](https://www.npmjs.com/package/d3-shape/v/3.2.0) | 3.2.0 | ISC |
| [d3-time](https://www.npmjs.com/package/d3-time/v/3.1.0) | 3.1.0 | ISC |
| [d3-time-format](https://www.npmjs.com/package/d3-time-format/v/4.1.0) | 4.1.0 | ISC |
| [d3-timer](https://www.npmjs.com/package/d3-timer/v/3.0.1) | 3.0.1 | ISC |
| [decimal.js-light](https://www.npmjs.com/package/decimal.js-light/v/2.5.1) | 2.5.1 | MIT |
| [dom-helpers](https://www.npmjs.com/package/dom-helpers/v/5.2.1) | 5.2.1 | MIT |
| [eventemitter3](https://www.npmjs.com/package/eventemitter3/v/4.0.7) | 4.0.7 | MIT |
| [fast-equals](https://www.npmjs.com/package/fast-equals/v/5.4.1) | 5.4.1 | MIT |
| [internmap](https://www.npmjs.com/package/internmap/v/2.0.3) | 2.0.3 | ISC |
| [js-tokens](https://www.npmjs.com/package/js-tokens/v/4.0.0) | 4.0.0 | MIT |
| [lodash](https://www.npmjs.com/package/lodash/v/4.18.1) | 4.18.1 | MIT |
| [loose-envify](https://www.npmjs.com/package/loose-envify/v/1.4.0) | 1.4.0 | MIT |
| [object-assign](https://www.npmjs.com/package/object-assign/v/4.1.1) | 4.1.1 | MIT |
| [prop-types](https://www.npmjs.com/package/prop-types/v/15.8.1) | 15.8.1 | MIT |
| [react](https://www.npmjs.com/package/react/v/18.3.1) | 18.3.1 | MIT |
| [react-dom](https://www.npmjs.com/package/react-dom/v/18.3.1) | 18.3.1 | MIT |
| [react-is](https://www.npmjs.com/package/react-is/v/16.13.1) | 16.13.1 | MIT |
| [react-is](https://www.npmjs.com/package/react-is/v/18.3.1) | 18.3.1 | MIT |
| [react-smooth](https://www.npmjs.com/package/react-smooth/v/4.0.4) | 4.0.4 | MIT |
| [react-transition-group](https://www.npmjs.com/package/react-transition-group/v/4.4.5) | 4.4.5 | BSD-3-Clause |
| [recharts](https://www.npmjs.com/package/recharts/v/2.15.4) | 2.15.4 | MIT |
| [recharts-scale](https://www.npmjs.com/package/recharts-scale/v/0.4.5) | 0.4.5 | MIT |
| [scheduler](https://www.npmjs.com/package/scheduler/v/0.23.2) | 0.23.2 | MIT |
| [tiny-invariant](https://www.npmjs.com/package/tiny-invariant/v/1.3.3) | 1.3.3 | MIT |
| [topojson-client](https://www.npmjs.com/package/topojson-client/v/3.1.0) | 3.1.0 | ISC |
| [use-sync-external-store](https://www.npmjs.com/package/use-sync-external-store/v/1.6.0) | 1.6.0 | MIT |
| [victory-vendor](https://www.npmjs.com/package/victory-vendor/v/36.9.2) | 36.9.2 | MIT AND ISC |
| [zustand](https://www.npmjs.com/package/zustand/v/4.5.7) | 4.5.7 | MIT |

### Build and test tools only

These are used to build and test the dashboard and are not shipped to visitors.

<details>
<summary>343 packages</summary>

| Package | Version | Licence |
|---|---|---|
| [@adobe/css-tools](https://www.npmjs.com/package/@adobe/css-tools/v/4.5.0) | 4.5.0 | MIT |
| [@asamuzakjp/css-color](https://www.npmjs.com/package/@asamuzakjp/css-color/v/3.2.0) | 3.2.0 | MIT |
| [@babel/code-frame](https://www.npmjs.com/package/@babel/code-frame/v/7.29.7) | 7.29.7 | MIT |
| [@babel/compat-data](https://www.npmjs.com/package/@babel/compat-data/v/7.29.7) | 7.29.7 | MIT |
| [@babel/core](https://www.npmjs.com/package/@babel/core/v/7.29.7) | 7.29.7 | MIT |
| [@babel/generator](https://www.npmjs.com/package/@babel/generator/v/7.29.8) | 7.29.8 | MIT |
| [@babel/helper-compilation-targets](https://www.npmjs.com/package/@babel/helper-compilation-targets/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-globals](https://www.npmjs.com/package/@babel/helper-globals/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-module-imports](https://www.npmjs.com/package/@babel/helper-module-imports/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-module-transforms](https://www.npmjs.com/package/@babel/helper-module-transforms/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-plugin-utils](https://www.npmjs.com/package/@babel/helper-plugin-utils/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-string-parser](https://www.npmjs.com/package/@babel/helper-string-parser/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-validator-identifier](https://www.npmjs.com/package/@babel/helper-validator-identifier/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helper-validator-option](https://www.npmjs.com/package/@babel/helper-validator-option/v/7.29.7) | 7.29.7 | MIT |
| [@babel/helpers](https://www.npmjs.com/package/@babel/helpers/v/7.29.7) | 7.29.7 | MIT |
| [@babel/parser](https://www.npmjs.com/package/@babel/parser/v/7.29.8) | 7.29.8 | MIT |
| [@babel/plugin-transform-react-jsx-self](https://www.npmjs.com/package/@babel/plugin-transform-react-jsx-self/v/7.29.7) | 7.29.7 | MIT |
| [@babel/plugin-transform-react-jsx-source](https://www.npmjs.com/package/@babel/plugin-transform-react-jsx-source/v/7.29.7) | 7.29.7 | MIT |
| [@babel/template](https://www.npmjs.com/package/@babel/template/v/7.29.7) | 7.29.7 | MIT |
| [@babel/traverse](https://www.npmjs.com/package/@babel/traverse/v/7.29.8) | 7.29.8 | MIT |
| [@babel/types](https://www.npmjs.com/package/@babel/types/v/7.29.8) | 7.29.8 | MIT |
| [@csstools/color-helpers](https://www.npmjs.com/package/@csstools/color-helpers/v/5.1.0) | 5.1.0 | MIT-0 |
| [@csstools/css-calc](https://www.npmjs.com/package/@csstools/css-calc/v/2.1.4) | 2.1.4 | MIT |
| [@csstools/css-color-parser](https://www.npmjs.com/package/@csstools/css-color-parser/v/3.1.0) | 3.1.0 | MIT |
| [@csstools/css-parser-algorithms](https://www.npmjs.com/package/@csstools/css-parser-algorithms/v/3.0.5) | 3.0.5 | MIT |
| [@csstools/css-tokenizer](https://www.npmjs.com/package/@csstools/css-tokenizer/v/3.0.4) | 3.0.4 | MIT |
| [@esbuild/aix-ppc64](https://www.npmjs.com/package/@esbuild/aix-ppc64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/android-arm](https://www.npmjs.com/package/@esbuild/android-arm/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/android-arm64](https://www.npmjs.com/package/@esbuild/android-arm64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/android-x64](https://www.npmjs.com/package/@esbuild/android-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/darwin-arm64](https://www.npmjs.com/package/@esbuild/darwin-arm64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/darwin-x64](https://www.npmjs.com/package/@esbuild/darwin-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/freebsd-arm64](https://www.npmjs.com/package/@esbuild/freebsd-arm64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/freebsd-x64](https://www.npmjs.com/package/@esbuild/freebsd-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-arm](https://www.npmjs.com/package/@esbuild/linux-arm/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-arm64](https://www.npmjs.com/package/@esbuild/linux-arm64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-ia32](https://www.npmjs.com/package/@esbuild/linux-ia32/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-loong64](https://www.npmjs.com/package/@esbuild/linux-loong64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-mips64el](https://www.npmjs.com/package/@esbuild/linux-mips64el/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-ppc64](https://www.npmjs.com/package/@esbuild/linux-ppc64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-riscv64](https://www.npmjs.com/package/@esbuild/linux-riscv64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-s390x](https://www.npmjs.com/package/@esbuild/linux-s390x/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/linux-x64](https://www.npmjs.com/package/@esbuild/linux-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/netbsd-x64](https://www.npmjs.com/package/@esbuild/netbsd-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/openbsd-x64](https://www.npmjs.com/package/@esbuild/openbsd-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/sunos-x64](https://www.npmjs.com/package/@esbuild/sunos-x64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/win32-arm64](https://www.npmjs.com/package/@esbuild/win32-arm64/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/win32-ia32](https://www.npmjs.com/package/@esbuild/win32-ia32/v/0.21.5) | 0.21.5 | MIT |
| [@esbuild/win32-x64](https://www.npmjs.com/package/@esbuild/win32-x64/v/0.21.5) | 0.21.5 | MIT |
| [@eslint-community/eslint-utils](https://www.npmjs.com/package/@eslint-community/eslint-utils/v/4.10.1) | 4.10.1 | MIT |
| [@eslint-community/regexpp](https://www.npmjs.com/package/@eslint-community/regexpp/v/4.12.2) | 4.12.2 | MIT |
| [@eslint/eslintrc](https://www.npmjs.com/package/@eslint/eslintrc/v/2.1.4) | 2.1.4 | MIT |
| [@eslint/js](https://www.npmjs.com/package/@eslint/js/v/8.57.1) | 8.57.1 | MIT |
| [@humanwhocodes/config-array](https://www.npmjs.com/package/@humanwhocodes/config-array/v/0.13.0) | 0.13.0 | Apache-2.0 |
| [@humanwhocodes/module-importer](https://www.npmjs.com/package/@humanwhocodes/module-importer/v/1.0.1) | 1.0.1 | Apache-2.0 |
| [@humanwhocodes/object-schema](https://www.npmjs.com/package/@humanwhocodes/object-schema/v/2.0.3) | 2.0.3 | BSD-3-Clause |
| [@jridgewell/gen-mapping](https://www.npmjs.com/package/@jridgewell/gen-mapping/v/0.3.13) | 0.3.13 | MIT |
| [@jridgewell/remapping](https://www.npmjs.com/package/@jridgewell/remapping/v/2.3.5) | 2.3.5 | MIT |
| [@jridgewell/resolve-uri](https://www.npmjs.com/package/@jridgewell/resolve-uri/v/3.1.2) | 3.1.2 | MIT |
| [@jridgewell/sourcemap-codec](https://www.npmjs.com/package/@jridgewell/sourcemap-codec/v/1.5.5) | 1.5.5 | MIT |
| [@jridgewell/trace-mapping](https://www.npmjs.com/package/@jridgewell/trace-mapping/v/0.3.31) | 0.3.31 | MIT |
| [@napi-rs/lzma-linux-x64-gnu](https://www.npmjs.com/package/@napi-rs/lzma-linux-x64-gnu/v/1.5.1) | 1.5.1 | MIT |
| [@nodelib/fs.scandir](https://www.npmjs.com/package/@nodelib/fs.scandir/v/2.1.5) | 2.1.5 | MIT |
| [@nodelib/fs.stat](https://www.npmjs.com/package/@nodelib/fs.stat/v/2.0.5) | 2.0.5 | MIT |
| [@nodelib/fs.walk](https://www.npmjs.com/package/@nodelib/fs.walk/v/1.2.8) | 1.2.8 | MIT |
| [@rolldown/pluginutils](https://www.npmjs.com/package/@rolldown/pluginutils/v/1.0.0-beta.27) | 1.0.0-beta.27 | MIT |
| [@rollup/rollup-android-arm-eabi](https://www.npmjs.com/package/@rollup/rollup-android-arm-eabi/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-android-arm64](https://www.npmjs.com/package/@rollup/rollup-android-arm64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-darwin-arm64](https://www.npmjs.com/package/@rollup/rollup-darwin-arm64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-darwin-x64](https://www.npmjs.com/package/@rollup/rollup-darwin-x64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-freebsd-arm64](https://www.npmjs.com/package/@rollup/rollup-freebsd-arm64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-freebsd-x64](https://www.npmjs.com/package/@rollup/rollup-freebsd-x64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-arm-gnueabihf](https://www.npmjs.com/package/@rollup/rollup-linux-arm-gnueabihf/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-arm-musleabihf](https://www.npmjs.com/package/@rollup/rollup-linux-arm-musleabihf/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-arm64-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-arm64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-arm64-musl](https://www.npmjs.com/package/@rollup/rollup-linux-arm64-musl/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-loong64-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-loong64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-loong64-musl](https://www.npmjs.com/package/@rollup/rollup-linux-loong64-musl/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-ppc64-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-ppc64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-ppc64-musl](https://www.npmjs.com/package/@rollup/rollup-linux-ppc64-musl/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-riscv64-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-riscv64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-riscv64-musl](https://www.npmjs.com/package/@rollup/rollup-linux-riscv64-musl/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-s390x-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-s390x-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-x64-gnu](https://www.npmjs.com/package/@rollup/rollup-linux-x64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-linux-x64-musl](https://www.npmjs.com/package/@rollup/rollup-linux-x64-musl/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-openbsd-x64](https://www.npmjs.com/package/@rollup/rollup-openbsd-x64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-openharmony-arm64](https://www.npmjs.com/package/@rollup/rollup-openharmony-arm64/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-win32-arm64-msvc](https://www.npmjs.com/package/@rollup/rollup-win32-arm64-msvc/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-win32-ia32-msvc](https://www.npmjs.com/package/@rollup/rollup-win32-ia32-msvc/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-win32-x64-gnu](https://www.npmjs.com/package/@rollup/rollup-win32-x64-gnu/v/4.63.0) | 4.63.0 | MIT |
| [@rollup/rollup-win32-x64-msvc](https://www.npmjs.com/package/@rollup/rollup-win32-x64-msvc/v/4.63.0) | 4.63.0 | MIT |
| [@testing-library/dom](https://www.npmjs.com/package/@testing-library/dom/v/10.4.2) | 10.4.2 | MIT |
| [@testing-library/jest-dom](https://www.npmjs.com/package/@testing-library/jest-dom/v/6.9.1) | 6.9.1 | MIT |
| [@testing-library/react](https://www.npmjs.com/package/@testing-library/react/v/16.3.3) | 16.3.3 | MIT |
| [@types/aria-query](https://www.npmjs.com/package/@types/aria-query/v/5.0.4) | 5.0.4 | MIT |
| [@types/babel__core](https://www.npmjs.com/package/@types/babel__core/v/7.20.5) | 7.20.5 | MIT |
| [@types/babel__generator](https://www.npmjs.com/package/@types/babel__generator/v/7.27.0) | 7.27.0 | MIT |
| [@types/babel__template](https://www.npmjs.com/package/@types/babel__template/v/7.4.4) | 7.4.4 | MIT |
| [@types/babel__traverse](https://www.npmjs.com/package/@types/babel__traverse/v/7.28.0) | 7.28.0 | MIT |
| [@types/d3-geo](https://www.npmjs.com/package/@types/d3-geo/v/3.1.1) | 3.1.1 | MIT |
| [@types/estree](https://www.npmjs.com/package/@types/estree/v/1.0.9) | 1.0.9 | MIT |
| [@types/geojson](https://www.npmjs.com/package/@types/geojson/v/7946.0.16) | 7946.0.16 | MIT |
| [@types/prop-types](https://www.npmjs.com/package/@types/prop-types/v/15.7.15) | 15.7.15 | MIT |
| [@types/react](https://www.npmjs.com/package/@types/react/v/18.3.31) | 18.3.31 | MIT |
| [@types/react-dom](https://www.npmjs.com/package/@types/react-dom/v/18.3.7) | 18.3.7 | MIT |
| [@types/topojson-client](https://www.npmjs.com/package/@types/topojson-client/v/3.1.5) | 3.1.5 | MIT |
| [@types/topojson-specification](https://www.npmjs.com/package/@types/topojson-specification/v/1.0.5) | 1.0.5 | MIT |
| [@typescript-eslint/eslint-plugin](https://www.npmjs.com/package/@typescript-eslint/eslint-plugin/v/7.18.0) | 7.18.0 | MIT |
| [@typescript-eslint/parser](https://www.npmjs.com/package/@typescript-eslint/parser/v/7.18.0) | 7.18.0 | BSD-2-Clause |
| [@typescript-eslint/scope-manager](https://www.npmjs.com/package/@typescript-eslint/scope-manager/v/7.18.0) | 7.18.0 | MIT |
| [@typescript-eslint/type-utils](https://www.npmjs.com/package/@typescript-eslint/type-utils/v/7.18.0) | 7.18.0 | MIT |
| [@typescript-eslint/types](https://www.npmjs.com/package/@typescript-eslint/types/v/7.18.0) | 7.18.0 | MIT |
| [@typescript-eslint/typescript-estree](https://www.npmjs.com/package/@typescript-eslint/typescript-estree/v/7.18.0) | 7.18.0 | BSD-2-Clause |
| [@typescript-eslint/utils](https://www.npmjs.com/package/@typescript-eslint/utils/v/7.18.0) | 7.18.0 | MIT |
| [@typescript-eslint/visitor-keys](https://www.npmjs.com/package/@typescript-eslint/visitor-keys/v/7.18.0) | 7.18.0 | MIT |
| [@ungap/structured-clone](https://www.npmjs.com/package/@ungap/structured-clone/v/1.4.0) | 1.4.0 | ISC |
| [@vitejs/plugin-react](https://www.npmjs.com/package/@vitejs/plugin-react/v/4.7.0) | 4.7.0 | MIT |
| [@vitest/expect](https://www.npmjs.com/package/@vitest/expect/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/mocker](https://www.npmjs.com/package/@vitest/mocker/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/pretty-format](https://www.npmjs.com/package/@vitest/pretty-format/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/runner](https://www.npmjs.com/package/@vitest/runner/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/snapshot](https://www.npmjs.com/package/@vitest/snapshot/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/spy](https://www.npmjs.com/package/@vitest/spy/v/2.1.9) | 2.1.9 | MIT |
| [@vitest/utils](https://www.npmjs.com/package/@vitest/utils/v/2.1.9) | 2.1.9 | MIT |
| [acorn](https://www.npmjs.com/package/acorn/v/8.18.0) | 8.18.0 | MIT |
| [acorn-jsx](https://www.npmjs.com/package/acorn-jsx/v/5.3.2) | 5.3.2 | MIT |
| [agent-base](https://www.npmjs.com/package/agent-base/v/7.1.4) | 7.1.4 | MIT |
| [ajv](https://www.npmjs.com/package/ajv/v/6.15.0) | 6.15.0 | MIT |
| [ansi-regex](https://www.npmjs.com/package/ansi-regex/v/5.0.1) | 5.0.1 | MIT |
| [ansi-styles](https://www.npmjs.com/package/ansi-styles/v/4.3.0) | 4.3.0 | MIT |
| [ansi-styles](https://www.npmjs.com/package/ansi-styles/v/5.2.0) | 5.2.0 | MIT |
| [argparse](https://www.npmjs.com/package/argparse/v/2.0.1) | 2.0.1 | Python-2.0 |
| [aria-query](https://www.npmjs.com/package/aria-query/v/5.3.0) | 5.3.0 | Apache-2.0 |
| [array-union](https://www.npmjs.com/package/array-union/v/2.1.0) | 2.1.0 | MIT |
| [assertion-error](https://www.npmjs.com/package/assertion-error/v/2.0.1) | 2.0.1 | MIT |
| [asynckit](https://www.npmjs.com/package/asynckit/v/0.4.0) | 0.4.0 | MIT |
| [balanced-match](https://www.npmjs.com/package/balanced-match/v/1.0.2) | 1.0.2 | MIT |
| [baseline-browser-mapping](https://www.npmjs.com/package/baseline-browser-mapping/v/2.11.19) | 2.11.19 | Apache-2.0 |
| [brace-expansion](https://www.npmjs.com/package/brace-expansion/v/1.1.18) | 1.1.18 | MIT |
| [brace-expansion](https://www.npmjs.com/package/brace-expansion/v/2.1.4) | 2.1.4 | MIT |
| [braces](https://www.npmjs.com/package/braces/v/3.0.3) | 3.0.3 | MIT |
| [browserslist](https://www.npmjs.com/package/browserslist/v/4.28.8) | 4.28.8 | MIT |
| [cac](https://www.npmjs.com/package/cac/v/6.7.14) | 6.7.14 | MIT |
| [call-bind-apply-helpers](https://www.npmjs.com/package/call-bind-apply-helpers/v/1.0.2) | 1.0.2 | MIT |
| [callsites](https://www.npmjs.com/package/callsites/v/3.1.0) | 3.1.0 | MIT |
| [caniuse-lite](https://www.npmjs.com/package/caniuse-lite/v/1.0.30001810) | 1.0.30001810 | CC-BY-4.0 |
| [chai](https://www.npmjs.com/package/chai/v/5.3.3) | 5.3.3 | MIT |
| [chalk](https://www.npmjs.com/package/chalk/v/4.1.2) | 4.1.2 | MIT |
| [check-error](https://www.npmjs.com/package/check-error/v/2.1.3) | 2.1.3 | MIT |
| [color-convert](https://www.npmjs.com/package/color-convert/v/2.0.1) | 2.0.1 | MIT |
| [color-name](https://www.npmjs.com/package/color-name/v/1.1.4) | 1.1.4 | MIT |
| [combined-stream](https://www.npmjs.com/package/combined-stream/v/1.0.8) | 1.0.8 | MIT |
| [concat-map](https://www.npmjs.com/package/concat-map/v/0.0.1) | 0.0.1 | MIT |
| [convert-source-map](https://www.npmjs.com/package/convert-source-map/v/2.0.0) | 2.0.0 | MIT |
| [cross-spawn](https://www.npmjs.com/package/cross-spawn/v/7.0.6) | 7.0.6 | MIT |
| [css.escape](https://www.npmjs.com/package/css.escape/v/1.5.1) | 1.5.1 | MIT |
| [cssstyle](https://www.npmjs.com/package/cssstyle/v/4.6.0) | 4.6.0 | MIT |
| [data-urls](https://www.npmjs.com/package/data-urls/v/5.0.0) | 5.0.0 | MIT |
| [debug](https://www.npmjs.com/package/debug/v/4.4.3) | 4.4.3 | MIT |
| [decimal.js](https://www.npmjs.com/package/decimal.js/v/10.6.0) | 10.6.0 | MIT |
| [deep-eql](https://www.npmjs.com/package/deep-eql/v/5.0.2) | 5.0.2 | MIT |
| [deep-is](https://www.npmjs.com/package/deep-is/v/0.1.4) | 0.1.4 | MIT |
| [delayed-stream](https://www.npmjs.com/package/delayed-stream/v/1.0.0) | 1.0.0 | MIT |
| [dequal](https://www.npmjs.com/package/dequal/v/2.0.3) | 2.0.3 | MIT |
| [dir-glob](https://www.npmjs.com/package/dir-glob/v/3.0.1) | 3.0.1 | MIT |
| [doctrine](https://www.npmjs.com/package/doctrine/v/3.0.0) | 3.0.0 | Apache-2.0 |
| [dom-accessibility-api](https://www.npmjs.com/package/dom-accessibility-api/v/0.5.16) | 0.5.16 | MIT |
| [dom-accessibility-api](https://www.npmjs.com/package/dom-accessibility-api/v/0.6.3) | 0.6.3 | MIT |
| [dunder-proto](https://www.npmjs.com/package/dunder-proto/v/1.0.1) | 1.0.1 | MIT |
| [electron-to-chromium](https://www.npmjs.com/package/electron-to-chromium/v/1.5.415) | 1.5.415 | ISC |
| [entities](https://www.npmjs.com/package/entities/v/6.0.1) | 6.0.1 | BSD-2-Clause |
| [es-define-property](https://www.npmjs.com/package/es-define-property/v/1.0.1) | 1.0.1 | MIT |
| [es-errors](https://www.npmjs.com/package/es-errors/v/1.3.0) | 1.3.0 | MIT |
| [es-module-lexer](https://www.npmjs.com/package/es-module-lexer/v/1.7.0) | 1.7.0 | MIT |
| [es-object-atoms](https://www.npmjs.com/package/es-object-atoms/v/1.1.2) | 1.1.2 | MIT |
| [es-set-tostringtag](https://www.npmjs.com/package/es-set-tostringtag/v/2.1.0) | 2.1.0 | MIT |
| [esbuild](https://www.npmjs.com/package/esbuild/v/0.21.5) | 0.21.5 | MIT |
| [escalade](https://www.npmjs.com/package/escalade/v/3.2.0) | 3.2.0 | MIT |
| [escape-string-regexp](https://www.npmjs.com/package/escape-string-regexp/v/4.0.0) | 4.0.0 | MIT |
| [eslint](https://www.npmjs.com/package/eslint/v/8.57.1) | 8.57.1 | MIT |
| [eslint-plugin-react-hooks](https://www.npmjs.com/package/eslint-plugin-react-hooks/v/4.6.2) | 4.6.2 | MIT |
| [eslint-plugin-react-refresh](https://www.npmjs.com/package/eslint-plugin-react-refresh/v/0.4.26) | 0.4.26 | MIT |
| [eslint-scope](https://www.npmjs.com/package/eslint-scope/v/7.2.2) | 7.2.2 | BSD-2-Clause |
| [eslint-visitor-keys](https://www.npmjs.com/package/eslint-visitor-keys/v/3.4.3) | 3.4.3 | Apache-2.0 |
| [espree](https://www.npmjs.com/package/espree/v/9.6.1) | 9.6.1 | BSD-2-Clause |
| [esquery](https://www.npmjs.com/package/esquery/v/1.7.0) | 1.7.0 | BSD-3-Clause |
| [esrecurse](https://www.npmjs.com/package/esrecurse/v/4.3.0) | 4.3.0 | BSD-2-Clause |
| [estraverse](https://www.npmjs.com/package/estraverse/v/5.3.0) | 5.3.0 | BSD-2-Clause |
| [estree-walker](https://www.npmjs.com/package/estree-walker/v/3.0.3) | 3.0.3 | MIT |
| [esutils](https://www.npmjs.com/package/esutils/v/2.0.3) | 2.0.3 | BSD-2-Clause |
| [expect-type](https://www.npmjs.com/package/expect-type/v/1.4.0) | 1.4.0 | Apache-2.0 |
| [fast-deep-equal](https://www.npmjs.com/package/fast-deep-equal/v/3.1.3) | 3.1.3 | MIT |
| [fast-glob](https://www.npmjs.com/package/fast-glob/v/3.3.3) | 3.3.3 | MIT |
| [fast-json-stable-stringify](https://www.npmjs.com/package/fast-json-stable-stringify/v/2.1.0) | 2.1.0 | MIT |
| [fast-levenshtein](https://www.npmjs.com/package/fast-levenshtein/v/2.0.6) | 2.0.6 | MIT |
| [fastq](https://www.npmjs.com/package/fastq/v/1.20.1) | 1.20.1 | ISC |
| [file-entry-cache](https://www.npmjs.com/package/file-entry-cache/v/6.0.1) | 6.0.1 | MIT |
| [fill-range](https://www.npmjs.com/package/fill-range/v/7.1.1) | 7.1.1 | MIT |
| [find-up](https://www.npmjs.com/package/find-up/v/5.0.0) | 5.0.0 | MIT |
| [flat-cache](https://www.npmjs.com/package/flat-cache/v/3.2.0) | 3.2.0 | MIT |
| [flatted](https://www.npmjs.com/package/flatted/v/3.4.4) | 3.4.4 | ISC |
| [form-data](https://www.npmjs.com/package/form-data/v/4.0.6) | 4.0.6 | MIT |
| [fs.realpath](https://www.npmjs.com/package/fs.realpath/v/1.0.0) | 1.0.0 | ISC |
| [fsevents](https://www.npmjs.com/package/fsevents/v/2.3.3) | 2.3.3 | MIT |
| [function-bind](https://www.npmjs.com/package/function-bind/v/1.1.2) | 1.1.2 | MIT |
| [gensync](https://www.npmjs.com/package/gensync/v/1.0.0-beta.2) | 1.0.0-beta.2 | MIT |
| [get-intrinsic](https://www.npmjs.com/package/get-intrinsic/v/1.3.0) | 1.3.0 | MIT |
| [get-proto](https://www.npmjs.com/package/get-proto/v/1.0.1) | 1.0.1 | MIT |
| [glob](https://www.npmjs.com/package/glob/v/7.2.3) | 7.2.3 | ISC |
| [glob-parent](https://www.npmjs.com/package/glob-parent/v/5.1.2) | 5.1.2 | ISC |
| [glob-parent](https://www.npmjs.com/package/glob-parent/v/6.0.2) | 6.0.2 | ISC |
| [globals](https://www.npmjs.com/package/globals/v/13.24.0) | 13.24.0 | MIT |
| [globby](https://www.npmjs.com/package/globby/v/11.1.0) | 11.1.0 | MIT |
| [gopd](https://www.npmjs.com/package/gopd/v/1.2.0) | 1.2.0 | MIT |
| [graphemer](https://www.npmjs.com/package/graphemer/v/1.4.0) | 1.4.0 | MIT |
| [has-flag](https://www.npmjs.com/package/has-flag/v/4.0.0) | 4.0.0 | MIT |
| [has-symbols](https://www.npmjs.com/package/has-symbols/v/1.1.0) | 1.1.0 | MIT |
| [has-tostringtag](https://www.npmjs.com/package/has-tostringtag/v/1.0.2) | 1.0.2 | MIT |
| [hasown](https://www.npmjs.com/package/hasown/v/2.0.4) | 2.0.4 | MIT |
| [html-encoding-sniffer](https://www.npmjs.com/package/html-encoding-sniffer/v/4.0.0) | 4.0.0 | MIT |
| [http-proxy-agent](https://www.npmjs.com/package/http-proxy-agent/v/7.0.2) | 7.0.2 | MIT |
| [https-proxy-agent](https://www.npmjs.com/package/https-proxy-agent/v/7.0.6) | 7.0.6 | MIT |
| [iconv-lite](https://www.npmjs.com/package/iconv-lite/v/0.6.3) | 0.6.3 | MIT |
| [ignore](https://www.npmjs.com/package/ignore/v/5.3.2) | 5.3.2 | MIT |
| [import-fresh](https://www.npmjs.com/package/import-fresh/v/3.3.1) | 3.3.1 | MIT |
| [imurmurhash](https://www.npmjs.com/package/imurmurhash/v/0.1.4) | 0.1.4 | MIT |
| [indent-string](https://www.npmjs.com/package/indent-string/v/4.0.0) | 4.0.0 | MIT |
| [inflight](https://www.npmjs.com/package/inflight/v/1.0.6) | 1.0.6 | ISC |
| [inherits](https://www.npmjs.com/package/inherits/v/2.0.4) | 2.0.4 | ISC |
| [is-extglob](https://www.npmjs.com/package/is-extglob/v/2.1.1) | 2.1.1 | MIT |
| [is-glob](https://www.npmjs.com/package/is-glob/v/4.0.3) | 4.0.3 | MIT |
| [is-number](https://www.npmjs.com/package/is-number/v/7.0.0) | 7.0.0 | MIT |
| [is-path-inside](https://www.npmjs.com/package/is-path-inside/v/3.0.3) | 3.0.3 | MIT |
| [is-potential-custom-element-name](https://www.npmjs.com/package/is-potential-custom-element-name/v/1.0.1) | 1.0.1 | MIT |
| [isexe](https://www.npmjs.com/package/isexe/v/2.0.0) | 2.0.0 | ISC |
| [js-yaml](https://www.npmjs.com/package/js-yaml/v/4.3.2) | 4.3.2 | MIT |
| [jsdom](https://www.npmjs.com/package/jsdom/v/25.0.1) | 25.0.1 | MIT |
| [jsesc](https://www.npmjs.com/package/jsesc/v/3.1.0) | 3.1.0 | MIT |
| [json-buffer](https://www.npmjs.com/package/json-buffer/v/3.0.1) | 3.0.1 | MIT |
| [json-schema-traverse](https://www.npmjs.com/package/json-schema-traverse/v/0.4.1) | 0.4.1 | MIT |
| [json-stable-stringify-without-jsonify](https://www.npmjs.com/package/json-stable-stringify-without-jsonify/v/1.0.1) | 1.0.1 | MIT |
| [json5](https://www.npmjs.com/package/json5/v/2.2.3) | 2.2.3 | MIT |
| [keyv](https://www.npmjs.com/package/keyv/v/4.5.4) | 4.5.4 | MIT |
| [levn](https://www.npmjs.com/package/levn/v/0.4.1) | 0.4.1 | MIT |
| [locate-path](https://www.npmjs.com/package/locate-path/v/6.0.0) | 6.0.0 | MIT |
| [lodash.merge](https://www.npmjs.com/package/lodash.merge/v/4.6.2) | 4.6.2 | MIT |
| [loupe](https://www.npmjs.com/package/loupe/v/3.2.1) | 3.2.1 | MIT |
| [lru-cache](https://www.npmjs.com/package/lru-cache/v/10.4.3) | 10.4.3 | ISC |
| [lru-cache](https://www.npmjs.com/package/lru-cache/v/5.1.1) | 5.1.1 | ISC |
| [lz-string](https://www.npmjs.com/package/lz-string/v/1.5.0) | 1.5.0 | MIT |
| [magic-string](https://www.npmjs.com/package/magic-string/v/0.30.21) | 0.30.21 | MIT |
| [math-intrinsics](https://www.npmjs.com/package/math-intrinsics/v/1.1.0) | 1.1.0 | MIT |
| [merge2](https://www.npmjs.com/package/merge2/v/1.4.1) | 1.4.1 | MIT |
| [micromatch](https://www.npmjs.com/package/micromatch/v/4.0.8) | 4.0.8 | MIT |
| [mime-db](https://www.npmjs.com/package/mime-db/v/1.52.0) | 1.52.0 | MIT |
| [mime-types](https://www.npmjs.com/package/mime-types/v/2.1.35) | 2.1.35 | MIT |
| [min-indent](https://www.npmjs.com/package/min-indent/v/1.0.1) | 1.0.1 | MIT |
| [minimatch](https://www.npmjs.com/package/minimatch/v/3.1.5) | 3.1.5 | ISC |
| [minimatch](https://www.npmjs.com/package/minimatch/v/9.0.9) | 9.0.9 | ISC |
| [ms](https://www.npmjs.com/package/ms/v/2.1.3) | 2.1.3 | MIT |
| [nanoid](https://www.npmjs.com/package/nanoid/v/3.3.18) | 3.3.18 | MIT |
| [natural-compare](https://www.npmjs.com/package/natural-compare/v/1.4.0) | 1.4.0 | MIT |
| [node-releases](https://www.npmjs.com/package/node-releases/v/2.0.53) | 2.0.53 | MIT |
| [nwsapi](https://www.npmjs.com/package/nwsapi/v/2.2.27) | 2.2.27 | MIT |
| [once](https://www.npmjs.com/package/once/v/1.4.0) | 1.4.0 | ISC |
| [optionator](https://www.npmjs.com/package/optionator/v/0.9.4) | 0.9.4 | MIT |
| [p-limit](https://www.npmjs.com/package/p-limit/v/3.1.0) | 3.1.0 | MIT |
| [p-locate](https://www.npmjs.com/package/p-locate/v/5.0.0) | 5.0.0 | MIT |
| [parent-module](https://www.npmjs.com/package/parent-module/v/1.0.1) | 1.0.1 | MIT |
| [parse5](https://www.npmjs.com/package/parse5/v/7.3.0) | 7.3.0 | MIT |
| [path-exists](https://www.npmjs.com/package/path-exists/v/4.0.0) | 4.0.0 | MIT |
| [path-is-absolute](https://www.npmjs.com/package/path-is-absolute/v/1.0.1) | 1.0.1 | MIT |
| [path-key](https://www.npmjs.com/package/path-key/v/3.1.1) | 3.1.1 | MIT |
| [path-type](https://www.npmjs.com/package/path-type/v/4.0.0) | 4.0.0 | MIT |
| [pathe](https://www.npmjs.com/package/pathe/v/1.1.2) | 1.1.2 | MIT |
| [pathval](https://www.npmjs.com/package/pathval/v/2.0.1) | 2.0.1 | MIT |
| [picocolors](https://www.npmjs.com/package/picocolors/v/1.1.1) | 1.1.1 | ISC |
| [picomatch](https://www.npmjs.com/package/picomatch/v/2.3.2) | 2.3.2 | MIT |
| [postcss](https://www.npmjs.com/package/postcss/v/8.5.26) | 8.5.26 | MIT |
| [prelude-ls](https://www.npmjs.com/package/prelude-ls/v/1.2.1) | 1.2.1 | MIT |
| [pretty-format](https://www.npmjs.com/package/pretty-format/v/27.5.1) | 27.5.1 | MIT |
| [punycode](https://www.npmjs.com/package/punycode/v/2.3.1) | 2.3.1 | MIT |
| [queue-microtask](https://www.npmjs.com/package/queue-microtask/v/1.2.3) | 1.2.3 | MIT |
| [react-is](https://www.npmjs.com/package/react-is/v/17.0.2) | 17.0.2 | MIT |
| [react-refresh](https://www.npmjs.com/package/react-refresh/v/0.17.0) | 0.17.0 | MIT |
| [redent](https://www.npmjs.com/package/redent/v/3.0.0) | 3.0.0 | MIT |
| [resolve-from](https://www.npmjs.com/package/resolve-from/v/4.0.0) | 4.0.0 | MIT |
| [reusify](https://www.npmjs.com/package/reusify/v/1.1.0) | 1.1.0 | MIT |
| [rimraf](https://www.npmjs.com/package/rimraf/v/3.0.2) | 3.0.2 | ISC |
| [rollup](https://www.npmjs.com/package/rollup/v/4.63.0) | 4.63.0 | MIT |
| [rrweb-cssom](https://www.npmjs.com/package/rrweb-cssom/v/0.7.1) | 0.7.1 | MIT |
| [rrweb-cssom](https://www.npmjs.com/package/rrweb-cssom/v/0.8.0) | 0.8.0 | MIT |
| [run-parallel](https://www.npmjs.com/package/run-parallel/v/1.2.0) | 1.2.0 | MIT |
| [safer-buffer](https://www.npmjs.com/package/safer-buffer/v/2.1.2) | 2.1.2 | MIT |
| [saxes](https://www.npmjs.com/package/saxes/v/6.0.0) | 6.0.0 | ISC |
| [semver](https://www.npmjs.com/package/semver/v/6.3.1) | 6.3.1 | ISC |
| [semver](https://www.npmjs.com/package/semver/v/7.8.5) | 7.8.5 | ISC |
| [shebang-command](https://www.npmjs.com/package/shebang-command/v/2.0.0) | 2.0.0 | MIT |
| [shebang-regex](https://www.npmjs.com/package/shebang-regex/v/3.0.0) | 3.0.0 | MIT |
| [siginfo](https://www.npmjs.com/package/siginfo/v/2.0.0) | 2.0.0 | ISC |
| [slash](https://www.npmjs.com/package/slash/v/3.0.0) | 3.0.0 | MIT |
| [source-map-js](https://www.npmjs.com/package/source-map-js/v/1.2.1) | 1.2.1 | BSD-3-Clause |
| [stackback](https://www.npmjs.com/package/stackback/v/0.0.2) | 0.0.2 | MIT |
| [std-env](https://www.npmjs.com/package/std-env/v/3.10.0) | 3.10.0 | MIT |
| [strip-ansi](https://www.npmjs.com/package/strip-ansi/v/6.0.1) | 6.0.1 | MIT |
| [strip-indent](https://www.npmjs.com/package/strip-indent/v/3.0.0) | 3.0.0 | MIT |
| [strip-json-comments](https://www.npmjs.com/package/strip-json-comments/v/3.1.1) | 3.1.1 | MIT |
| [supports-color](https://www.npmjs.com/package/supports-color/v/7.2.0) | 7.2.0 | MIT |
| [symbol-tree](https://www.npmjs.com/package/symbol-tree/v/3.2.4) | 3.2.4 | MIT |
| [text-table](https://www.npmjs.com/package/text-table/v/0.2.0) | 0.2.0 | MIT |
| [tinybench](https://www.npmjs.com/package/tinybench/v/2.9.0) | 2.9.0 | MIT |
| [tinyexec](https://www.npmjs.com/package/tinyexec/v/0.3.2) | 0.3.2 | MIT |
| [tinypool](https://www.npmjs.com/package/tinypool/v/1.1.1) | 1.1.1 | MIT |
| [tinyrainbow](https://www.npmjs.com/package/tinyrainbow/v/1.2.0) | 1.2.0 | MIT |
| [tinyspy](https://www.npmjs.com/package/tinyspy/v/3.0.2) | 3.0.2 | MIT |
| [tldts](https://www.npmjs.com/package/tldts/v/6.1.86) | 6.1.86 | MIT |
| [tldts-core](https://www.npmjs.com/package/tldts-core/v/6.1.86) | 6.1.86 | MIT |
| [to-regex-range](https://www.npmjs.com/package/to-regex-range/v/5.0.1) | 5.0.1 | MIT |
| [tough-cookie](https://www.npmjs.com/package/tough-cookie/v/5.1.2) | 5.1.2 | BSD-3-Clause |
| [tr46](https://www.npmjs.com/package/tr46/v/5.1.1) | 5.1.1 | MIT |
| [ts-api-utils](https://www.npmjs.com/package/ts-api-utils/v/1.4.3) | 1.4.3 | MIT |
| [type-check](https://www.npmjs.com/package/type-check/v/0.4.0) | 0.4.0 | MIT |
| [type-fest](https://www.npmjs.com/package/type-fest/v/0.20.2) | 0.20.2 | (MIT OR CC0-1.0) |
| [typescript](https://www.npmjs.com/package/typescript/v/5.9.3) | 5.9.3 | Apache-2.0 |
| [update-browserslist-db](https://www.npmjs.com/package/update-browserslist-db/v/1.3.1) | 1.3.1 | MIT |
| [uri-js](https://www.npmjs.com/package/uri-js/v/4.4.1) | 4.4.1 | BSD-2-Clause |
| [vite](https://www.npmjs.com/package/vite/v/5.4.21) | 5.4.21 | MIT |
| [vite-node](https://www.npmjs.com/package/vite-node/v/2.1.9) | 2.1.9 | MIT |
| [vitest](https://www.npmjs.com/package/vitest/v/2.1.9) | 2.1.9 | MIT |
| [w3c-xmlserializer](https://www.npmjs.com/package/w3c-xmlserializer/v/5.0.0) | 5.0.0 | MIT |
| [webidl-conversions](https://www.npmjs.com/package/webidl-conversions/v/7.0.0) | 7.0.0 | BSD-2-Clause |
| [whatwg-encoding](https://www.npmjs.com/package/whatwg-encoding/v/3.1.1) | 3.1.1 | MIT |
| [whatwg-mimetype](https://www.npmjs.com/package/whatwg-mimetype/v/4.0.0) | 4.0.0 | MIT |
| [whatwg-url](https://www.npmjs.com/package/whatwg-url/v/14.2.0) | 14.2.0 | MIT |
| [which](https://www.npmjs.com/package/which/v/2.0.2) | 2.0.2 | ISC |
| [why-is-node-running](https://www.npmjs.com/package/why-is-node-running/v/2.3.0) | 2.3.0 | MIT |
| [word-wrap](https://www.npmjs.com/package/word-wrap/v/1.2.5) | 1.2.5 | MIT |
| [wrappy](https://www.npmjs.com/package/wrappy/v/1.0.2) | 1.0.2 | ISC |
| [ws](https://www.npmjs.com/package/ws/v/8.21.3) | 8.21.3 | MIT |
| [xml-name-validator](https://www.npmjs.com/package/xml-name-validator/v/5.0.0) | 5.0.0 | Apache-2.0 |
| [xmlchars](https://www.npmjs.com/package/xmlchars/v/2.2.0) | 2.2.0 | MIT |
| [yallist](https://www.npmjs.com/package/yallist/v/3.1.1) | 3.1.1 | ISC |
| [yocto-queue](https://www.npmjs.com/package/yocto-queue/v/0.1.0) | 0.1.0 | MIT |

</details>
