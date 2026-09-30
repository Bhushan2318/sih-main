"""Regenerate the third-party licence files from the project's real dependency sets.

    pip install uv remotezip           # once; neither is a project dependency
    (cd frontend && npm ci)            # the JavaScript licence texts are read from node_modules
    python backend/scripts/gen_third_party_notices.py

Writes, at the repository root:

  * THIRD_PARTY_NOTICES.md - every data source, Python and JavaScript package, font and
    container image, with its licence and the attribution it asks for;
  * THIRD_PARTY_LICENSES.txt - the full licence text of every package shipped to users: the
    serving image's Python packages and the site's JavaScript bundle;
  * frontend/public/third-party-licenses.txt - the JavaScript part, which the site serves.

How each fact is obtained, so the files can be trusted rather than hoped for:

  * The Python sets are resolved with `uv pip compile` for Linux x86_64 and Python 3.12, the
    Docker image's platform. pip's own resolver evaluates environment markers for the
    machine it runs on, so on a Mac it adds Python-3.9 backports and drops Linux-only
    packages such as PyTorch's NVIDIA libraries; uv evaluates them for the target.
  * Licences come from each release's PyPI metadata, normalised to SPDX, with the
    OVERRIDE entries checked by hand against the package's own licence file.
  * Licence texts are read out of the exact Linux wheels with HTTP range requests, so the
    ~500 MB of wheels are never downloaded. Small wheels that refuse ranges are fetched
    whole. Two releases ship no licence file in the wheel; theirs is read from the source
    repository at the release tag (UPSTREAM).
  * The JavaScript set is frontend/package-lock.json; "dev" entries are build tools and are
    not shipped. Texts are read from frontend/node_modules.

The data-source, font and container-image sections are written by hand below. Their terms
are quoted in LICENSES/, fetched from each provider.
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
DATE = datetime.date.today().isoformat()
SETS = {"serving": [], "dev": ["requirements-dev.txt"], "live": ["requirements-live.txt"],
        "train": ["requirements-train.txt"]}
LIC_FILE = re.compile(r"\.dist-info/(licenses/.+|(LICEN[CS]E|COPYING|NOTICE|AUTHORS)[^/]*)$", re.I)
UPSTREAM = {"pydantic-core": "https://raw.githubusercontent.com/pydantic/pydantic-core/v{v}/LICENSE",
            "xgboost": "https://raw.githubusercontent.com/dmlc/xgboost/v{v}/LICENSE"}
JS_UPSTREAM = {"victory-vendor": "https://raw.githubusercontent.com/FormidableLabs/victory/main/LICENSE.txt"}
JS_LIC = re.compile(r"^(licen[cs]e|copying|notice)(\.|-|$)", re.I)


def _get(url: str, timeout: int = 120) -> bytes:
    return urllib.request.urlopen(url, timeout=timeout).read()


def _pypi(name: str, version: str) -> dict:
    return json.loads(_get(f"https://pypi.org/pypi/{name}/{version}/json", 60))


def resolve(work: Path) -> None:
    """uv_<set>.txt pins and <set>.json metadata for every set, as the renderer reads them."""
    for s, extra in SETS.items():
        reqs = [str(BACKEND / "requirements.txt")] + [str(BACKEND / f) for f in extra]
        subprocess.run(["uv", "pip", "compile", "--quiet", "--no-header", "--no-annotate",
                        "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28",
                        *reqs, "-o", str(work / f"uv_{s}.txt")], check=True)
        install = []
        for line in open(work / f"uv_{s}.txt"):
            if "==" not in line:
                continue
            n, v = [x.strip() for x in line.split(";")[0].split("==")]
            info = _pypi(n, v)["info"]
            install.append({"metadata": {"name": info["name"], "version": v,
                                         "license": info.get("license"),
                                         "license_expression": info.get("license_expression"),
                                         "classifier": info.get("classifiers", [])}})
        json.dump({"install": install}, open(work / f"{s}.json", "w"))
        print(f"resolved {s}: {len(install)} packages", flush=True)
    nv = {}
    for it in json.load(open(work / "train.json"))["install"]:
        m = it["metadata"]
        if m["name"].lower().startswith("nvidia-"):
            nv[m["name"]] = {"version": m["version"]}
    json.dump(nv, open(work / "nvidia_meta.json", "w"))


def _pick_wheel(urls: list) -> dict | None:
    def score(f: str) -> int:
        if "cp312" in f and "manylinux" in f and "x86_64" in f: return 0
        if "abi3" in f and "manylinux" in f and "x86_64" in f: return 1
        if "none-any" in f: return 2
        if "manylinux" in f and "x86_64" in f: return 3
        return 9
    whl = sorted((u for u in urls if u["filename"].endswith(".whl")), key=lambda u: score(u["filename"]))
    return whl[0] if whl and score(whl[0]["filename"]) < 9 else None


def serving_texts(work: Path) -> None:
    from remotezip import RemoteZip
    out = []
    for line in open(work / "uv_serving.txt"):
        if "==" not in line:
            continue
        name, ver = [x.strip() for x in line.split(";")[0].split("==")]
        info = _pypi(name, ver)
        w = _pick_wheel(info["urls"])
        files, src = {}, None
        if w:
            try:
                with RemoteZip(w["url"]) as z:
                    files = {n.split(".dist-info/")[1]: z.read(n).decode("utf-8", "replace")
                             for n in z.namelist() if LIC_FILE.search(n)}
            except Exception:  # small wheels refuse range requests: fetch them whole
                z = zipfile.ZipFile(io.BytesIO(_get(w["url"])))
                files = {n.split(".dist-info/")[1]: z.read(n).decode("utf-8", "replace")
                         for n in z.namelist() if LIC_FILE.search(n)}
        key = name.lower().replace("_", "-")
        if not files and key in UPSTREAM:
            src = UPSTREAM[key].format(v=ver)
            files = {"LICENSE (from the source repository at the release tag)": _get(src).decode()}
        if not files:
            sys.exit(f"no licence text found for {name}=={ver}: add it to UPSTREAM")
        out.append({"name": name, "version": ver, "wheel": w["filename"] if w else None,
                    "files": files, "files_source": src})
        print(f"licence text: {name}=={ver} ({len(files)} file(s))", flush=True)
    json.dump(out, open(work / "serving_licences.json", "w"))


def js_texts(work: Path) -> Path:
    lock = json.load(open(REPO / "frontend/package-lock.json"))
    out = []
    for path, d in lock["packages"].items():
        if not path or d.get("dev") or d.get("devOptional"):
            continue
        name = path.split("node_modules/")[-1]
        pdir = REPO / "frontend" / path
        if not pdir.is_dir():
            sys.exit(f"{pdir} is missing: run `npm ci` in frontend/ first")
        files = {f.name: f.read_text(errors="replace") for f in sorted(pdir.iterdir())
                 if f.is_file() and JS_LIC.match(f.name)}
        vendored = pdir / "lib-vendor"
        if not files and vendored.is_dir():
            for mod in sorted(vendored.iterdir()):
                if (mod / "LICENSE").is_file():
                    files[f"lib-vendor/{mod.name}/LICENSE (vendored module)"] = (mod / "LICENSE").read_text()
        if name in JS_UPSTREAM:
            files[f"{JS_UPSTREAM[name].rsplit('/', 1)[1]} (upstream repository, main branch)"] = \
                _get(JS_UPSTREAM[name]).decode()
        if not files:
            sys.exit(f"no licence text found for {name}@{d.get('version')}: add it to JS_UPSTREAM")
        out.append({"name": name, "version": d.get("version"), "license": d.get("license"), "files": files})
    p = work / "js_runtime_licences.json"
    json.dump(out, open(p, "w"))
    return p


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        resolve(work)
        serving_texts(work)
        js = js_texts(work)
        render(work, REPO, js)


def render(REP: Path, REPO: Path, JS: Path) -> None:
    def pins(f):
        out = {}
        for l in open(REP / f):
            if "==" in l:
                n, v = [x.strip() for x in l.split(";")[0].split("==")]
                out[n.lower().replace("_", "-")] = (n, v)
        return out
    S = {k: pins(f"uv_{k}.txt") for k in ("serving", "dev", "live", "train")}

    meta = {}
    for s in ("serving", "dev", "live", "train"):
        for it in json.load(open(REP / f"{s}.json"))["install"]:
            m = it["metadata"]; meta[m["name"].lower().replace("_", "-")] = m
    nv = json.load(open(REP / "nvidia_meta.json"))

    NORM = {"mit": "MIT", "mit license": "MIT", "bsd-3-clause": "BSD-3-Clause", "bsd 3-clause license": "BSD-3-Clause",
            "bsd 3-clause": "BSD-3-Clause", "new bsd": "BSD-3-Clause", "3-clause bsd license": "BSD-3-Clause",
            "bsd-3": "BSD-3-Clause", "apache 2.0": "Apache-2.0", "apache-2.0": "Apache-2.0",
            "apache license version 2.0": "Apache-2.0", "apache license, version 2.0": "Apache-2.0",
            "apache license v2.0": "Apache-2.0", "mpl-2.0": "MPL-2.0", "mpl-2.0 and mit": "MPL-2.0 AND MIT",
            "mit or apache-2.0": "MIT OR Apache-2.0"}
    CLS = {"MIT License": "MIT", "Apache Software License": "Apache-2.0", "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
           "BSD License": "BSD-3-Clause"}
    # Checked against each package's own licence file (or, for packages not shipped, its project).
    OVERRIDE = {"numpy": "BSD-3-Clause", "scipy": "BSD-3-Clause", "numba": "BSD-2-Clause",
                "python-dateutil": "Apache-2.0 AND BSD-3-Clause", "uvloop": "MIT OR Apache-2.0",
                "uvicorn": "BSD-3-Clause", "kiwisolver": "BSD-3-Clause", "cycler": "BSD-3-Clause",
                "matplotlib": "LicenseRef-Matplotlib (PSF-based)", "mpmath": "BSD-3-Clause", "sympy": "BSD-3-Clause",
                "jinja2": "BSD-3-Clause", "eccodeslib": "Apache-2.0", "eckitlib": "Apache-2.0"}

    def pylic(k):
        if k in OVERRIDE: return OVERRIDE[k]
        if k in nv: return "LicenseRef-NVIDIA-Proprietary"
        m = meta.get(k, {})
        if m.get("license_expression"): return m["license_expression"]
        lic = (m.get("license") or "").strip()
        if lic.lower() in NORM: return NORM[lic.lower()]
        cls = [c.split("::")[-1].strip() for c in m.get("classifier", []) if c.startswith("License ::")]
        for c in cls:
            if c in CLS: return CLS[c]
        return lic.split("\n")[0][:60] or "see project"

    def pytable(keys, src):
        rows = ["| Package | Version | Licence |", "|---|---|---|"]
        for k in sorted(keys):
            n, v = src[k]
            rows.append(f"| [{n}](https://pypi.org/project/{n}/{v}/) | {v} | {pylic(k)} |")
        return "\n".join(rows)

    serv = S["serving"]
    extra = {s: {k: S[s][k] for k in S[s] if k not in serv} for s in ("dev", "live", "train")}

    lock = json.load(open(REPO / "frontend/package-lock.json"))
    js_all = []
    for path, d in lock["packages"].items():
        if not path: continue
        js_all.append((path.split("node_modules/")[-1], d.get("version"), d.get("license") or "see package",
                       bool(d.get("dev") or d.get("devOptional"))))
    js_rt = sorted({(n, v, l) for n, v, l, dev in js_all if not dev})
    js_dev = sorted({(n, v, l) for n, v, l, dev in js_all if dev})
    def jstable(rows):
        out = ["| Package | Version | Licence |", "|---|---|---|"]
        out += [f"| [{n}](https://www.npmjs.com/package/{n}/v/{v}) | {v} | {l} |" for n, v, l in rows]
        return "\n".join(out)

    cnt_serv = collections.Counter(pylic(k) for k in serv)
    cnt_js = collections.Counter(l for _, _, l in js_rt)
    def counts(c): return ", ".join(f"{k} ({v})" for k, v in c.most_common())

    notices = f"""# Third-party notices

Sanket's own code is released under the [MIT Licence](LICENSE). This file lists everything
else the project is built on: the data, the Python and JavaScript libraries, the fonts and
the container images. For each it gives the licence and any attribution the source asks for.

- [`REUSE.toml`](REUSE.toml) records the copyright and licence of every file in this
  repository, and `reuse lint` checks it ([REUSE 3.3](https://reuse.software)).
- [`LICENSES/`](LICENSES/) holds the full text of every licence that applies to a file here.
- [`THIRD_PARTY_LICENSES.txt`](THIRD_PARTY_LICENSES.txt) holds the full licence text of every
  package shipped to users: the {len(serv)} Python packages in the serving image and the
  {len(js_rt)} JavaScript packages bundled into the site. The site serves the JavaScript part
  itself at [`/third-party-licenses.txt`](https://sanket-a0dd.onrender.com/third-party-licenses.txt).

Generated {DATE}. The Python sets were resolved with `uv pip compile` for Linux x86_64 and
Python 3.12, the platform of the Docker image. The licence texts were read from those exact
wheels. The JavaScript set comes from `frontend/package-lock.json`.

## Summary

| What | Count | Licences |
|---|---|---|
| Data sources | 10 | see [Data](#data) |
| Python packages in the serving image | {len(serv)} | {counts(cnt_serv)} |
| Python packages added for tests | {len(extra['dev'])} | see [Python packages](#python-packages) |
| Python packages added for live ingestion | {len(extra['live'])} | see [Python packages](#python-packages) |
| Python packages added for CNN training | {len(extra['train'])} | see [Python packages](#python-packages) |
| JavaScript packages bundled into the site | {len(js_rt)} | {counts(cnt_js)} |
| JavaScript build and test tools | {len(js_dev)} | see [JavaScript packages](#javascript-packages) |
| Web fonts | 3 | OFL-1.1 |

## Data

Every value Sanket shows comes from one of these. The terms quoted in `LICENSES/` were
fetched from each provider on {DATE}.

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

{pytable(serv.keys(), serv)}

### Added for tests (`requirements-dev.txt`)

{pytable(extra['dev'].keys(), extra['dev'])}

### Added for live ingestion (`requirements-live.txt`)

{pytable(extra['live'].keys(), extra['live'])}

### Added for CNN training (`requirements-train.txt`)

On Linux x86_64, PyTorch from PyPI also installs NVIDIA's CUDA libraries (the `nvidia-*`
packages below), which are proprietary software under
[NVIDIA's licence](https://docs.nvidia.com/cuda/eula/index.html). They are needed only to
train the convolutional challenger on a GPU, and never enter the serving image.

{pytable(extra['train'].keys(), extra['train'])}

## JavaScript packages

### Bundled into the site

{jstable(js_rt)}

### Build and test tools only

These are used to build and test the dashboard and are not shipped to visitors.

<details>
<summary>{len(js_dev)} packages</summary>

{jstable(js_dev)}

</details>
"""
    (REPO / "THIRD_PARTY_NOTICES.md").write_text(notices)

    rule = "-" * 78
    def block(title, src, text):
        return f"{rule}\n{title}\n{src}\n{rule}\n\n{text.strip()}\n\n"
    py = json.load(open(REP / "serving_licences.json"))
    js = json.load(open(JS))
    head = f"""Third-party licence texts
=========================

The full licence text of every third-party package shipped to users of Sanket, generated
{DATE}:

  * the {len(py)} Python packages in the serving image (backend/requirements.txt, resolved
    for Linux x86_64 and Python 3.12), read from the exact wheels that image installs;
  * the {len(js)} JavaScript packages bundled into the site, read from
    frontend/node_modules after `npm ci` against frontend/package-lock.json.

Sanket's own code is under the MIT licence (LICENSE). THIRD_PARTY_NOTICES.md lists every
other dependency, data source, font and image, with its licence.

"""
    parts_py, parts_js = [], []
    for r in py:
        k = r["name"].lower().replace("_", "-")
        for fname, text in r["files"].items():
            src = f"Source: {r['wheel']}, {fname}" if not r.get("files_source") else f"Source: {r['files_source']}"
            parts_py.append(block(f"Python: {r['name']} {r['version']} ({pylic(k)})", src, text))
    for r in sorted(js, key=lambda x: (x["name"], x["version"])):
        for fname, text in r["files"].items():
            parts_js.append(block(f"JavaScript: {r['name']} {r['version']} ({r['license']})", f"Source: {r['name']}/{fname}", text))
    (REPO / "THIRD_PARTY_LICENSES.txt").write_text(head + "PYTHON PACKAGES (serving image)\n\n" + "".join(parts_py) + "JAVASCRIPT PACKAGES (site bundle)\n\n" + "".join(parts_js))
    site_head = f"""Third-party licence texts for the Sanket website
==================================================

The site's JavaScript bundle contains the {len(js)} open-source packages below. Their licences
follow, generated {DATE} from frontend/package-lock.json. Sanket's own code is under the MIT
licence. Every other dependency, data source and font is listed with its licence in
THIRD_PARTY_NOTICES.md at https://github.com/Bhushan2318/sih-main

"""
    (REPO / "frontend/public/third-party-licenses.txt").write_text(site_head + "".join(parts_js))
    for f in ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.txt", "frontend/public/third-party-licenses.txt"):
        p = REPO / f; print(f, p.stat().st_size, "bytes")
    print("serving licences:", counts(cnt_serv)); print("site bundle licences:", counts(cnt_js))
    print("extras:", {k: len(v) for k, v in extra.items()}, "| js runtime", len(js_rt), "| js dev", len(js_dev))


if __name__ == "__main__":
    main()
