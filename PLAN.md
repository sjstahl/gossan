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

---

## 12. Status (October 2026) and how the build diverged from the plan

The phases above were built in order, then extended. This section is the current truth; sections 1 to 11 are the original intent.

**Built and tested (441+ tests):**

| Plan | State |
|---|---|
| 1 Foundation, 2 BC harvest | Done and much wider: **25 BC sources** (occurrences, tenure, ARIS, NoW, RGS; bedrock geology, faults, Quaternary cover, mineral potential; MINFILE production and inventory; historic notices, mine permits, crown grants; parks, conservancies, reserves, treaty land; roads, power lines; NRCan MRDEM-30 terrain). Identical re-downloads are not stored twice (`core/rawstore.py`). |
| 3 Leads | Done (BC and US scoring, drilling indicator). Leads now carry target overlap, the model's prospectivity percentile, a combined score and the tenure watch/change flags (below). |
| 4 Digitize | Done through review and drillhole tables; the tables now feed the analysis, decision table and (optionally) the model (below). Never exercised on real approved data. |
| 5 Spectral | **Not built.** Only scene search and download (EMIT, EnMAP, ASTER) exist; `src/gossan/spectral/` is empty. |
| 6 Targeting | Done and reworked (below). |
| 7 Report | Pitch pack PDF, short Word report, and the new **analysis report** (PDF and Word from one content model). |

**Added beyond the plan:**

- `gossan analyze` (`src/gossan/analyze/`): data inventory; RGS geochemistry (element quality tiers with below-detection handling, Fe-Mn correction, pathfinder indices, compositional PCA, robust multivariate anomalies); DEM stream catchments (own D8 routing, no GPL hydrology dependency); known occurrences against bedrock units and faults; per-target decision table (ground status, tenure, exclusions, access, history).
- Target model: evidence features (faults, contacts, intrusions, unit age, terrain, geochemistry, host-unit favourability) instead of exploration effort; occurrence classes; automatic CV buffer; effort-matched AUC; minimum target size and optional smoothing.
- US jurisdiction (MRDS, BLM claims, Nevada notices).
- Tenure change detection (`analyze/tenure.py`): the raw tenure snapshots Gossan already keeps are compared (new, gone, owner changed, transferred, work filed, cash in lieu, extended, reduced), plus a watchlist (expiring, recently staked or updated, frequently transferred) that works from one harvest; shown in the analysis report, the decision table, and QGIS-ready layers.
- Digitize-to-target loop: approved collar/assay tables are placed, unit-converted and classified (`analyze/drill_results.py`: mineralized, barren, inconclusive); the decision table says what drilling already tested each target; `target.drill_labels` (evaluate by default) reports AUC against known-barren drill cells, and `train` also uses barren holes as negatives and mineralized holes as positives (unvalidated on real data); `analyze --steps priorities` ranks drilling reports worth digitizing and `--pdf-order priority` downloads them first.
- `gossan run` (`src/gossan/pipeline.py`): harvest, target, analyze and report in dependency order, each step degrading on its own, with a status table and a provenance record of the run; also behind the QGIS tool's "Also analyze" option.
- Project settings are versioned (`config_version`); opening an older `project.yaml` migrates defaults that were deliberately changed (so a project created before the CV gap became automatic does not silently keep `cv_buffer_m: 0`).
- Provenance carries the code revision; `uv.lock` pins the tested dependency set; `scripts/backup_repo.py` writes verified git bundles; CI workflow in `.github/workflows/`.

**Findings that shaped the defaults (East Kootenay test area, 23 metallic occurrences; treat as indicative):**

- Using exploration effort (ARIS and tenure density) as features gives a high plain AUC (0.94) that mostly measures where people explored; against equally explored ground it is about 0.6, the same as the evidence-only model. Effort is off by default.
- Cross-validation without a gap between test blocks and training cells flatters the model: effort-matched AUC 0.63 with no gap, 0.56 at 2 km, 0.52 at 5 km, chance at 10 km. The model's skill is local (it extends known camps better than it finds new ones far from them).
- Correcting geochemistry for lithology or dilution lowered the indices' ability to pick out catchments with known occurrences, because host units carry the mineralization there. Both are opt-in.

- Regional magnetics (`target/magnetics.py`): a magnetic grid ingested as `magnetics` (`gossan ingest ... --buffer-m 5000 [--crs ...]`; GeoTIFF, ER Mapper, GXF, ASCII; Geosoft .grd must be exported) becomes six features: field, local anomaly (minus a 2 km upward continuation), first vertical derivative, analytic signal, tilt angle and texture (FFT, tapered padding, checked against exact analytic solutions). No public automatic download exists, so `harvest --source nrcan_geophysics` prints guided steps (BC compilation DOI 10.4095/p1xpddgswc). No reduction to the pole is applied. **The effect on model skill is unmeasured on real data** (no real grid was available); the report shows the field and tilt map, and figures are now numbered automatically.

- ASTER alteration indices (`spectral/aster.py`, `gossan spectral`, also run by `gossan run` before the model): five band ratios (ferric, gossan, ferrous, Al-OH, Mg-OH/carbonate) from the scenes in `raw/satellite/aster/`, on radiance (DN converted with the handbook coefficients; gain is a setting, not read from the files), masked for vegetation and snow, median across scenes, written as `rasters/aster_*.tif` (so the model picks them up when they cover enough of the AOI) with `analysis/spectral_summary.json` (coverage, and AUC of each index for metallic occurrences) and a report section. **Not run on real scenes**: downloading needs an Earthdata token and the download path has never been exercised live; the file naming is the connector's documented pattern, unverified. Expect most of a BC AOI to be masked by forest; ASTER SWIR scenes stop at April 2008. The spectral library work in 7.5 (continuum removal, sub-band fitting) is not built.

- BC leads combined with the model (`leads/combine.py`): each lead gets target overlap (% and ha, best target), the 90th-percentile prospectivity of its ground and that value's percentile in the AOI, the tenure watch flags and latest snapshot change, and `combined_score = (1 - target_weight) * score + target_weight * prospectivity percentile` (default weight 0.3) with `combined_rank`; the lead `score`/`rank` do not change. `leads run` warns when the model's AUC is near 0.5 (it was 0.55 on the test area), when the merit half is close to noise.

- Knowledge-driven score (`target/knowledge.py`, on by default, `target.knowledge` in `project.yaml`): a weighted mean of rule-based favourabilities (near faults, contacts and intrusives; named host units; the top of the pathfinder geochemistry, magnetic analytic signal and ASTER ratios), with weights from a deposit-model preset (starting points, not validated) that can be replaced one by one. It never uses the occurrences, so they check it independently: the same spatial folds and metrics as the forest, plus each layer's own AUC (the synthetic test shows a contradicted rule scoring 0.19, and the report prints that). Outputs `target/knowledge.tif`, `knowledge_targets.gpkg` (with overlap with forest targets) and `knowledge_report.json`; forest targets and the decision table gain `knowledge_pctile` and `knowledge_support` (supported / partly / forest only), and unsupported targets get a note. Host units named because occurrences sit in them make its check circular; the report says so. `target.knowledge.host_mode: empirical` instead learns the host layer from the occurrences (unit favourability, as the forest does) and relearns it inside every cross-validation fold from the training blocks only, with the CV gap, so the held-out figures are not circular (the all-cell AUC and the layer's own AUC are marked in-sample); the default stays `named`. On the four BC validation areas it raised the knowledge score's held-out AUC by 0.02 to 0.10 and effort-matched AUC by 0.01 to 0.05 (positive in every area, thin evidence with five folds each, and the score then depends on the occurrences: `docs/validation/host_mode_2026-10.md`). Not run on the real test area.

- Client package (`deliver/`, `gossan deliver`): `outputs/deliverable/` with a QGIS project (`.qgs` written as XML, relative paths, styled; loaded and rendered in a real QGIS 3.36 in the tests), `map.html` (one offline file: pan/zoom, layer toggles, raster opacity, click for attributes with the decision table merged into the targets, scale bar, latitude/longitude readout for UTM CRSs; no basemap by design), `data/deliverable.gpkg`, `rasters/`, the report, `README.txt` and `manifest.json` with file hashes. Every item goes in only if its licence is on record and not restrictive (Access Only, internal use only, no record, ARIS names are withheld and listed); leads are never packaged; `--include-internal` lifts the licence test and marks every face INTERNAL. In the BC test project the decision table and report come out withheld when ARIS is among their inputs, so the client package is then the targets, scores and open context layers only.

- Branded Word reports (`report/branding.py`, `branding:` in `project.yaml`, `gossan template`): company, logo (header and cover, with alt text), accent colour, fonts, footer text, contact, optional cover page, and an optional company `.docx` template whose styles, page setup, header and footer the report is built inside (missing styles are supplied; its body text is dropped). The draft/INTERNAL footer line and page number are always added to every footer, the cover's included, and the banner stays in the body, so branding cannot remove the Access Only marking. A missing or unreadable logo or template stops the Word report with a message. Applies to the analysis report and the simple site report; **PDF reports are not branded**, and the Word output was verified by reading the files back, not in Word (none was installed).

- Validation on larger BC areas (`scripts/validate_regions.py`, results in `docs/validation/`): four areas of 7,000 to 16,000 km² (Boundary, East Kootenay wide, Quesnel, Stewart). Forest effort-matched AUC 0.55 to 0.65; the unfitted knowledge score matched or beat it in three of four areas; metallic-only labels helped in three of four; harvest and all steps worked at up to 1.6 M cells (2.2 GB peak, ~20 min for two model runs); one bug found and fixed (decision table when no sample index is usable). Findings and limits in `docs/validation/bc_regions_2026-10.md`.

**Done:** the ten review items. Decisions from the validation: `target.occurrence_class` defaults to `auto` (metallic where deposit-type codes exist; settings version 3 migrates projects still on `all`), and the generic knowledge mix drops the intrusive rule (kept in the porphyry, vein and gold presets).

**Next (see the review of October 2026):** validation on larger BC areas.
