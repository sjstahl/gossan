# Gossan — Build Plan

> Working name. Gossan: the oxidized cap over a buried sulphide body — surface clues to what's hidden underneath.

Internal tool for **SJS Geospatial Analytics / SJS Exploration & Analytics**. Not a commercial product. It exists to make consulting jobs faster, repeatable, and better documented, and to help find clients.

---

## 1. Purpose

Gossan turns a claim polygon (or a pile of legacy files) into client-ready deliverables. It supports three lines of work:

| Service | What the client gets | Gossan modules |
|---|---|---|
| **Desktop study / data compilation** | Compiled public data, maps, report | core, harvest, report |
| **Legacy data digitization & migration** | Georeferenced maps, extracted drill/assay tables, report inventory, clean GIS + drillhole database | core, harvest, digitize, report |
| **Alteration mapping & targeting** | Mineral/alteration maps, ranked targets with uncertainty | spectral, target, report |
| **Business development (internal)** | Ranked list of properties likely to need digitization or compilation, plus pitch samples | leads, digitize |

## 2. Guiding principles

1. **Human-in-the-loop over full automation.** A geologist checks every client deliverable. Automation does the grunt work; review steps are built in, not bolted on.
2. **Every job is reproducible.** A project folder plus its `project.yaml` must be enough to rebuild every output.
3. **Provenance on everything.** Every dataset records source, URL/identifier, retrieval date, and license/terms. Reports cite from this log automatically.
4. **Never stall a job.** Every automated step has a manual fallback (e.g. QGIS GCPs when auto-georeferencing fails).
5. **BC first.** Build and validate against BC data sources; add Alberta and others later behind the same interfaces.
6. **Small, tested phases.** Each phase ends with a working CLI command and passing tests before the next begins.

## 3. Tech stack

- **Python 3.11+**, packaged with `pyproject.toml` (uv or pip)
- **CLI:** Typer
- **Config / validation:** Pydantic + YAML
- **Vector:** GeoPandas, Shapely, Fiona/pyogrio → **GeoPackage**
- **Raster:** rasterio, rioxarray, xarray → **Cloud Optimized GeoTIFF (COG)**
- **Georeferencing:** GDAL (transforms, warping)
- **PDF / OCR:** PyMuPDF, pdfplumber, EasyOCR (Tesseract as fallback), OpenCV
- **Table extraction from scans:** Claude API (vision) with strict JSON schema output
- **ML:** scikit-learn (+ PU-bagging, spatial CV)
- **Reports:** python-docx with templates, matplotlib for figures
- **Testing:** pytest, with small fixture datasets committed to `tests/fixtures/`
- **Lint/format:** ruff; type hints throughout
- **Compute:** run locally by default; optional AWS EC2 for heavy batch jobs

## 4. Repository layout

```
gossan/
├── PLAN.md
├── CLAUDE.md                # conventions for Claude Code (see §10)
├── pyproject.toml
├── src/gossan/
│   ├── cli.py               # Typer entry point
│   ├── core/                # config, AOI, CRS, cache, provenance, logging
│   ├── harvest/             # one connector per data source
│   │   ├── base.py          # Connector interface
│   │   └── bc/              # minfile.py, mto_tenure.py, aris.py, rgs.py, ...
│   │   └── national/        # nrcan_geophysics.py, ...
│   │   └── satellite/       # emit.py, enmap.py, aster.py
│   ├── digitize/            # text, tables, georef, catalogue, review
│   ├── leads/               # scoring, scheduling, sample packs
│   ├── spectral/            # thin wrapper around sjs-spectral
│   ├── target/              # prospectivity models
│   └── report/              # docx builder + templates
├── templates/               # Word templates, map styles
└── tests/
    └── fixtures/
```

**Shared spectral library:** pull the spectral functions shared with SJS Image Analytics (continuum removal, Al-OH fitting, USGS library handling, bandpass convolution) into a separate package, `sjs-spectral`, imported by both projects. Fix bugs in one place.

## 5. Project folder (one per client job)

```
<project>/
├── project.yaml             # AOI, CRS, client, enabled modules, parameters
├── aoi.gpkg                 # claim polygon(s)
├── data.gpkg                # all vector layers
├── rasters/                 # COGs
├── raw/                     # downloaded originals, untouched
├── digitize/                # page images, extracted tables, GCPs, review queue
├── outputs/                 # maps, figures, reports
└── provenance.jsonl         # one line per dataset/operation
```

## 6. Data standards

- **Vectors:** GeoPackage, one layer per dataset, project CRS (UTM zone set from AOI).
- **Rasters:** COG, project CRS, nodata defined.
- **Digitized geology:** USGS **GeMS** schema (MapUnitPolys, ContactsAndFaults, DescriptionOfMapUnits, DataSources).
- **Drillholes:** `collar`, `survey`, `assay`, `lithology` tables with standard keys (`hole_id`, `from_m`, `to_m`); units and detection limits recorded explicitly.
- **Provenance record:** `{dataset, source, identifier, url, retrieved_at, license, operation, parameters, gossan_version}`.

## 7. Modules

### 7.1 core
- Load/validate `project.yaml` (Pydantic).
- AOI handling: read polygon, pick UTM zone, buffer options.
- Download cache keyed by source + query.
- Provenance logger used by every other module.

### 7.2 harvest
Common `Connector` interface: `search(aoi) → items`, `fetch(items) → files`, `load(files) → GeoDataFrame | raster`, each call logged to provenance.

BC connectors (Phase 2):
- MINFILE mineral occurrences
- Mineral Titles Online tenure (current + change history if available)
- ARIS assessment report index + PDFs
- Regional Geochemical Survey (RGS) stream sediment data

National:
- NRCan geophysics (magnetics, gravity, radiometrics grids)

Satellite (Phase 5):
- EMIT, EnMAP, ASTER

**Before writing each connector:** confirm the access method (API, WFS, bulk download, scraping) and terms of use. Record the terms in the connector docstring. Prefer official APIs/bulk downloads over scraping.

### 7.3 leads
- Score each tenure or property by: recency of tenure change × density/age of ARIS reports × commodity/deposit-type fit.
- Optional enrichment: owner company name, links to public announcements (manual review).
- Output: ranked CSV + map layer; `gossan leads run` suitable for a weekly schedule.
- **Sample pack generator:** pick 2–3 ARIS maps/tables for a lead property, run them through `digitize`, produce a short PDF/Word pitch showing before/after.

### 7.4 digitize (replaces DIGMAPPER)
Pipeline per report, with a manual fallback at every step:
1. **Text layer:** PyMuPDF extracts embedded text; pages without text go to OCR (EasyOCR → Tesseract fallback).
2. **Tables:** pdfplumber for digital tables; for scanned drill logs/assay sheets, send page images to the Claude API with a strict JSON schema matching the drillhole tables. All extracted tables go to a **review queue** (CSV side-by-side with page image) before entering the database.
3. **Georeferencing (semi-automatic):**
   - OpenCV detects grid ticks / graticule lines near map margins.
   - OCR reads the coordinate labels → candidate GCPs.
   - GDAL fits the transform; report residuals; reject if RMS above threshold.
   - **Fallback:** user picks GCPs in the QGIS Georeferencer; Gossan ingests the saved `.points` file.
4. **Catalogue:** index each report — year, company, commodities, work type, page/map/table counts — into `data.gpkg`. Feeds `leads` and gives clients a data inventory.
5. **Vectorization:** out of scope for automation for now. Manual QGIS work into GeMS; Gossan provides the empty GeMS template and validation.

DIGMAPPER may return later as an optional georeferencing backend behind the same interface; nothing else depends on it.

### 7.5 spectral
- Thin wrapper around `sjs-spectral`: continuum removal (convex hull), parabolic sub-band fitting for white-mica Al-OH position, USGS library matching, sensor bandpass convolution.
- Batch-run on harvested scenes with parameters fixed in `project.yaml`.
- Outputs: mineral/alteration COGs + QA figures.

### 7.6 target
- Feature stack from harvest + spectral outputs (rasterized to a common grid).
- Models: Random Forest baseline; positive-unlabeled bagging for sparse known occurrences.
- **Spatial cross-validation** (block CV), not random splits.
- Outputs: prospectivity map, uncertainty map, feature importance, ranked targets.

### 7.7 report
- python-docx builder using Word templates in `templates/`.
- Sections assembled from module outputs: data inventory, maps, tables, methods, references.
- References and data sources generated from `provenance.jsonl`.
- Drafts only — interpretation is written by the geologist.

## 8. CLI (target shape)

```
gossan new <name> --aoi claims.shp
gossan harvest <project> [--source minfile,aris,...]
gossan digitize <project> [--report <aris_id>]
gossan review <project>            # open the review queue
gossan spectral <project>
gossan target <project>
gossan report <project> --template desktop_study
gossan leads run [--region bc]
gossan leads pack <lead_id>
```

## 9. Build phases

Each phase ends when its **done-when** check passes and tests are green.

| Phase | Scope | Done when |
|---|---|---|
| **1. Foundation** | Repo, packaging, `core`, `gossan new`, MINFILE connector, minimal report | A claim polygon produces a project folder and a Word report with a MINFILE map and auto-generated data sources |
| **2. BC harvest** | MTO tenure, ARIS index + PDF download, RGS, NRCan geophysics | `gossan harvest` fills `data.gpkg` and `rasters/` for a test AOI, all with provenance |
| **3. Leads** | Scoring, ranked CSV + map, scheduled run | Weekly run produces a ranked lead list for a test region |
| **4. Digitize** | Text, tables (+ Claude API), semi-auto georef with QGIS fallback, catalogue, review queue, sample pack | One real ARIS report → georeferenced map, reviewed assay table in drillhole DB, catalogue entry, pitch sample |
| **5. Spectral** | `sjs-spectral` extraction, EMIT/EnMAP/ASTER connectors, batch processing | Alteration COGs for a test AOI match current notebook results |
| **6. Targeting** | Feature stack, RF + PU-bagging, spatial CV, uncertainty | Prospectivity + uncertainty maps with documented CV scores |
| **7. Later** | Alberta sources, ARIS full-text search, handheld SWIR interpreter, optional DIGMAPPER backend | — |

## 10. Conventions for Claude Code (copy into CLAUDE.md)

- Work one phase at a time; don't start the next until the done-when check passes.
- Every connector and pipeline step gets a pytest test against small fixtures in `tests/fixtures/` — no network calls in tests (mock or record responses).
- Type hints everywhere; ruff clean.
- Never write into `raw/`; originals are immutable.
- Every function that creates or transforms data writes a provenance record.
- No hardcoded paths, credentials, or API keys — config and environment variables only.
- Manual fallbacks are features, not afterthoughts: document them in the CLI help.
- Before adding a dependency, check its license; flag GPL/AGPL for review.

## 11. Open questions / risks

- **Data access & terms:** confirm access method and reuse terms for MTO, ARIS, MINFILE, RGS, and NRCan before building connectors.
- **Tenure change history:** confirm whether historical tenure events are available, or whether Gossan needs to snapshot tenure regularly to detect changes itself.
- **Claude API costs** for large scanned reports — batch, cache results, and only send pages flagged as tables.
- **Georef accuracy** on old maps (NAD27 vs NAD83, hand-drawn grids, distorted scans) — record datum explicitly; RMS thresholds in config.
- **Client confidentiality:** client-supplied legacy data stays local/private; never mixed into shared caches.
- **Scope creep:** vectorization and full NLP search stay in Phase 7 until digitization is earning.
