# Tutorial 3: Review and Reporting

Gossan automates the grunt work, but **a geologist signs off on everything that gets reported**. This tutorial covers the human-in-the-loop steps on the command line:

1. Digitize ARIS assessment reports (AI-read tables, auto-georeferenced maps).
2. Review each extracted item and **approve** or **reject** it with `gossan review`.
3. Build the PDF pitch pack with `gossan report`.
4. Handle ARIS **"Access Only"** data correctly before anything leaves the company.

**Before you start:** you have a project with harvested data ([Tutorial 2](02_QGIS_End_to_End_Workflow.md)). Scanned-table extraction also needs `OPENROUTER_API_KEY` ([Tutorial 1](01_Installation_and_Setup.md#5-set-api-keys-and-environment-variables)).

All commands below are run from the **base directory** that holds the project folder, with the Gossan environment active:

```powershell
C:\gossan\venv\Scripts\Activate.ps1
cd D:\GossanProjects
```

Replace `skeena_2026` with your project name. You can also pass a full path to the project folder.

---

## How review works

| Pipeline step | Creates | Review queue status |
|---|---|---|
| `gossan digitize ... --extract-scanned-tables` | A CSV per scanned table page, read by Claude vision | `PENDING` (type `table`) |
| `gossan digitize` (map pages) or `gossan ingest-points` | A georeferenced GeoTIFF per map page | `PENDING` (type `map`) |
| **You:** `gossan review approve` | Table rows loaded into the drillhole tables; map marked as checked | `APPROVED` |
| **You:** `gossan review reject` | Nothing loaded | `REJECTED` |

Key rules:

- **Nothing unreviewed is used.** Only `APPROVED` tables enter the drillhole database, and only `APPROVED` items appear in the pitch pack.
- **Decisions are final.** An approved or rejected item cannot be changed. Check carefully before you decide.
- **Every decision is logged** to `provenance.jsonl` with who decided, when, and your note.
- The queue is a plain CSV, `<project>/digitize/review_queue.csv`. You can open it in Excel to browse, but **change statuses only with `gossan review`**, so the decision is validated and logged.

---

## 1. Download and digitize ARIS reports

### Download report PDFs

```powershell
gossan harvest skeena_2026 --source aris_index --limit-pdfs 5
```

- Downloads up to 5 report PDFs (newest first) from the `aris_index` layer into `<project>/raw/aris_pdfs/`.
- Reports can be tens of MB each; start small.

### Catalogue, extract text and auto-georeference maps

```powershell
gossan digitize skeena_2026
```

This processes every downloaded PDF:

- Adds each report to the `catalogue` layer in `data.gpkg` (page count, likely map pages).
- Writes the text to `digitize/text/<aris_number>.txt` (OCR'd pages are marked `[OCR EXTRACTED]`).
- Extracts digital (text-based) tables to `digitize/tables/digital/`.
- Tries to georeference every page flagged as a map. Successes go to `rasters/aris_<aris_number>_p<page>.tif` and the review queue.

To limit the work to one report: `gossan digitize skeena_2026 --report 12345`.

### Read scanned tables with Claude vision

First find the pages that hold drill logs or assay sheets: open the PDF in `raw/aris_pdfs/` and note the page numbers. Then:

```powershell
gossan digitize skeena_2026 --report 12345 --target-pages 43-47 --extract-scanned-tables
```

- Only the pages you name are sent (`--target-pages` accepts `43-47` or `3, 43-47`); never more than **25 pages per run**.
- **This costs money** and sends the page images to OpenRouter and Anthropic. The run ends with a token count.
- Each table is written to `digitize/tables/scanned/<aris_number>_p<page>.csv` and queued as `PENDING`.
- Pages already extracted are skipped, so re-running does not pay twice.

---

## 2. List what is waiting for review

```powershell
gossan review list skeena_2026
```

(`gossan review skeena_2026` does the same.) Example output:

```text
3 item(s) pending review (1 map, 2 table)
  R0001  map    ARIS 12345 p.12  rasters/aris_12345_p12.tif  georeferenced automatically, 6 GCPs, RMS 8.2 m (limit 15 m, datum EPSG:26909); check placement against known features.
         image: digitize/maps/12345_p12.png
  R0002  table  ARIS 12345 p.44  digitize/tables/scanned/12345_p44.csv  assay table, 38 row(s), read by anthropic/claude-opus-5.5; check against the page image.
         image: digitize/pages/12345_p44.png
  R0003  table  ARIS 12345 p.45  digitize/tables/scanned/12345_p45.csv  assay table, 41 row(s), ...
         image: digitize/pages/12345_p45.png
```

Each line gives the **item ID** (`R0002`), the file to check, and the scan to check it against (`image:`).

---

## 3. Review an AI-extracted table

Claude's output always has the right *structure*, but the **values are a model's reading of a scan** and can be wrong.

### Compare the CSV with the scan

- Open the page image (for example `digitize/pages/12345_p44.png`) on one screen.
- Open the CSV (for example `digitize/tables/scanned/12345_p44.csv`) on the other. Use a text editor (VS Code, Notepad++), or import it into Excel with every column as **Text**.

> Opening a CSV directly in Excel can silently change values: dropped leading zeros in hole IDs, or `1-2` turned into a date. If you edit in Excel, import as Text and save as **CSV UTF-8**.

### Check, cell by cell

- **Digits and decimal points** in every assay value.
- **Column headings and units**, written as `Au (ppb)`, `Cu (ppm)`, `Lithology`.
- **Detection-limit flags** kept as printed (`<5`, `>10000`, `nil`). Gossan stores `<5` as value 5 with flag `<`, never as a plain 5.
- **`hole_id`** filled in on every row.
- **`from_m` / `to_m`** depths. If the log was in feet, Gossan has already converted to metres: spot-check a few intervals.
- **Illegible cells** marked `?`. Resolve every one from the scan, or blank the cell.
- **Merged or split rows** where the scan was smudged or the table broke across lines.

### Fix the CSV, then approve

Correct errors **directly in the CSV and save it**. The file is read at the moment you approve.

```powershell
gossan review approve skeena_2026 R0002 --by "S. Stahl" --note "Checked vs p.44; corrected Au in rows 12 and 30"
```

Expected output:

```text
Approved R0002 (table, ARIS 12345 p.44): 38 row(s) -> 152 record(s) in the assay table
```

- `--by` defaults to your Windows login name. `--note` is optional but recommended.
- The rows go into the matching table in `data.gpkg`: `assay`, `lithology`, `collar` or `survey`, one record per hole, interval and attribute. Open them in QGIS like any other layer.
- Rows with no depths are loaded as **surface samples** (soil, rock chips), and the output says how many.

### If approval fails validation

Nothing is loaded and the item **stays `PENDING`**. The error lists up to ten problems. Fix the CSV and run the same command again. The CSV must have:

- `hole_id`, `from_m` and `to_m` columns, plus at least one attribute column.
- Numeric, non-negative depths, with `to_m` greater than `from_m`.
- Both depths given, or both empty (surface sample).
- No `?` cells left.
- A table type of assay, lithology, collar or survey. A page Claude classed as `other` cannot be loaded: reject it.

### Reject a table

```powershell
gossan review reject skeena_2026 R0003 --note "Wrong table: claim summary, not assays"
```

To have Claude read the page again, delete `digitize/tables/scanned/<aris_number>_p<page>.csv` and re-run the `--extract-scanned-tables` command. This is charged again and creates a new review item.

---

## 4. Review a georeferenced historical map

### Check the placement in QGIS

- Add the map: drag `rasters/aris_12345_p12.tif` onto the canvas.
- Add reference layers:
  - A basemap: **Browser ▸ XYZ Tiles ▸ OpenStreetMap**, or satellite imagery.
  - From `data.gpkg`: `mto_tenure`, `minfile`.
- Make the map semi-transparent: right-click the layer ▸ **Properties ▸ Transparency**, set **Global Opacity** to about 50%.
- Compare features that appear on both: lakes, rivers, roads, claim corners, plotted showings.
- Note the **RMS** in the review note. A low RMS only means the control points agree with each other; the overlay check is what counts.
- A consistent shift of one to a few hundred metres usually means the **datum** is wrong. Old ARIS maps are often NAD27: in `project.yaml`, under `georef:`, set `datum: NAD27` before georeferencing further maps.

### Approve

```powershell
gossan review approve skeena_2026 R0001 --note "Lakes and road align within ~20 m"
```

Approving a map loads nothing; it marks the map as checked and makes it eligible for the pitch pack's *Before & After* section.

### Reject and fix with the QGIS Georeferencer

```powershell
gossan review reject skeena_2026 R0001 --note "Grid labels misread; ~300 m east offset"
```

Then georeference the page by hand:

- Open **Layer ▸ Georeferencer...**.
- **File ▸ Open Raster...** and choose `digitize/maps/12345_p12.png`. Use this exact image; do not crop or re-export it, because the point positions refer to its pixels.
- If `digitize/gcps/12345_p12_auto.points` exists, load Gossan's candidate points with **File ▸ Load GCP Points...** and correct them.
- Otherwise click **Add Point** and place **at least 4** control points on grid intersections, entering their coordinates.
- Set the CRS of the coordinates you entered in **Settings ▸ Transformation Settings ▸ Target CRS**.
- **File ▸ Save GCP Points As...**, for example `digitize/gcps/12345_p12_manual.points`.

Load the points into Gossan:

```powershell
gossan ingest-points skeena_2026 12345 12 skeena_2026\digitize\gcps\12345_p12_manual.points
```

- Arguments: project, ARIS number, page number, points file.
- Gossan fits and warps the map, checks the RMS against `georef.rms_threshold_m` (default 15 m), and queues a **new `PENDING` item**. Review it as above.
- For a map with known poor drafting, accept a looser fit for this map only with `--rms-threshold 25`; it is recorded in the review note.

Use the same route when `gossan digitize` reports `map p.N: not georeferenced`; its message gives the exact `ingest-points` command.

---

## 5. Build the PDF pitch pack

Run the steps in this order for the most complete pack:

```powershell
gossan harvest skeena_2026                 # property, tenure, ARIS index, geochemistry
gossan target skeena_2026                  # prospectivity map and targets
gossan leads run skeena_2026               # ranked tenures (optional)
gossan review list skeena_2026             # nothing important left PENDING
gossan report skeena_2026
```

Output: `<project>/outputs/report.pdf`. Figures are saved alongside it in `outputs/report_assets/`.

### Options

| Command | Result |
|---|---|
| `gossan report skeena_2026` | PDF pitch pack, `outputs/report.pdf`. |
| `gossan report skeena_2026 --lead 1234567` | Features one lead (a tenure number from `outputs/leads_ranked.csv`) with its ARIS reports. Needs `gossan leads run` first. |
| `gossan report skeena_2026 --format docx` | Short Word data-sources report, `outputs/report.docx`. |
| `gossan report skeena_2026 --format both` | Both files. |

### What is in the pack

| Section | Content | Appears when |
|---|---|---|
| Executive summary | Headline numbers from every step, with caveats. | Always |
| Property and tenure | Overview map, tenure summary, data inventory. | Always |
| Lead scoring | Ranked leads; the featured lead with `--lead`. | After `gossan leads run` |
| Prospectivity targeting | Score map with targets and occurrences, uncertainty map, target list, feature importance, spatial CV tables. | After `gossan target` |
| Before & After | Each **approved** map and table beside its original scan. | After approvals |
| Data sources and provenance | Every dataset with source, date and licence; licence notices. | Always |

A section without its data says which command produces it. If items are still pending, the *Before & After* section says how many. Rebuild the pack at any time; each build is logged with its checksum in `provenance.jsonl`.

### Before you share it

- Read the whole pack. Scores, maps and extracted tables are **machine-generated aids, not geological interpretation**.
- Confirm every *Before & After* item is one you are happy to show.
- Follow the data governance rules below. **The pack is laid out for clients, but it is marked INTERNAL DRAFT by design.**

---

## The analysis report (BC)

The pitch pack is about leads and digitized material. For a desktop study of a BC area, `gossan report --kind analysis` writes a separate report from what `gossan analyze` and `gossan target` produced.

```bash
gossan run skeena_2026 --class metallic      # all four steps, in order
```

`gossan run` is harvest (geology, faults, land status, roads, terrain, ...), then `target`, then `analyze`, then this report. The steps depend on each other in that order, which is why one command is safer than four. A step that cannot run does not stop the others; a table at the end lists each step as `ok`, `partial`, `skipped` (with the reason, for example too few known occurrences for a model) or `failed`, and the exit status is 0 only if every step was `ok`. To start from an AOI file: `gossan run skeena_2026 --aoi claims.gpkg`. To redo only the later steps from data you already have (offline): `gossan run skeena_2026 --steps target,analyze,report`. Or run the steps yourself:

```bash
gossan harvest skeena_2026
gossan target skeena_2026 --class metallic
gossan analyze skeena_2026
gossan report skeena_2026 --kind analysis
```

It writes `outputs/analysis_report.pdf` **and** `outputs/analysis_report.docx` with the same content (`--format pdf` or `--format docx` for one). The figures are saved in `outputs/report_assets/analysis/`.

| Section | What it shows |
|---|---|
| Summary and *Read this first* | Headline findings written from the numbers, and the caveats that apply (few occurrences, weak skill against equally explored ground, out-of-date analysis files). |
| Data compiled | Every dataset with its count, AOI coverage and licence. |
| Geology and known occurrences | Geology map with faults and occurrences, occurrences per bedrock unit against the unit's area, distance to faults. |
| Stream-sediment geochemistry | Catchment map, samples to follow up, elements used, main patterns. |
| Targets | Prospectivity map, what drives the model, cross-validation including the effort-matched AUC. |
| Target decision table | Ground status, host unit, geochemistry, road distance and notes per target, with a map. |
| Method and limits, Data sources | How it was made, and every dataset with its licence. |

A part whose inputs are missing says which command produces it. If the analysis files are older than `data.gpkg`, the summary says to re-run `gossan analyze`. **When any Access Only dataset is in the project (ARIS, roads, crown grants, ...) every page is marked INTERNAL DRAFT** and the banner names the datasets; the same governance rules below apply.

The Word file is for editing; it has the same headings, tables and figures, with figure captions in the *Caption* style.

### Tenure watch: what changed on the ground

Mineral Titles Online publishes only *current* tenure, so it cannot tell you what changed. Gossan keeps the history itself: every `gossan harvest` saves the raw tenure layer (an identical download is reused, not stored again) and logs when it looked, and `gossan analyze --steps tenure` compares consecutive saved snapshots. The report's **Tenure activity** section and `analysis/tenure_*.csv` show:

| Change | Meaning |
|---|---|
| `new` | not listed before: newly staked or registered |
| `gone` | no longer listed: lapsed, forfeited, converted or cancelled (the layer does not say which) |
| `owner changed`, `transferred` | the holders differ, or the ownership-transfer count rose |
| `work filed`, `cash in lieu` | assessment work was filed, or cash was paid instead: an active holder |
| `extended` | the good-to date moved later |
| `reduced`, `area changed` | the area shrank (or a reduction was recorded) or grew by more than 1% |

Each change says between which two dates it happened (when the old state was last seen and when the new one was first seen). A snapshot that suddenly lists no tenure at all is treated as a failed download and ignored, not as every claim lapsing.

The **watchlist** works from the very first harvest, from fields the layer already carries: tenures whose good-to date is within 180 days, tenures staked in the last year, records updated in the last 90 days, and tenures that have changed hands twice or more (change the thresholds under `tenure:` in `project.yaml`). Targets whose ground saw a change get a note in the decision table.

Changes only appear once at least two snapshots exist, so run the pipeline on a schedule. Weekly is plenty:

```powershell
gossan run skeena_2026 --steps harvest,analyze,report
```

On Windows, Task Scheduler can run that command weekly (**Create Basic Task**, trigger *Weekly*, action *Start a program*, program `C:\gossan\venv\Scripts\gossan.exe`, arguments as above, *Start in* your base folder). Each run adds a snapshot only if the tenure layer actually changed.

---

## Drilling indicator on BC leads

A plain `gossan harvest` on a BC project also downloads `drilling_notices`: Mines Act **Notice of Work** applications and permits that include surface exploration drilling. `gossan leads run` then adds, to every lead:

- **`drilling_flag`**: `active` (approved, permit open, not past its authorisation end date), `pending` (application in progress), `lapsed` (closed or past its end date) or `none`. Rejected and withdrawn applications are ignored. A notice counts for a tenure if it lists the tenure number or its point is within 500 m of the claim.
- **`drilling_points`**: a small bonus added to the score (default 10 active, 5 pending, 0 lapsed; the score is capped at 100). Set `drilling_active_points`, `drilling_pending_points`, `drilling_lapsed_points` or `drilling_match_buffer_m` under `parameters: leads:` in `project.yaml`; set the points to 0 to keep the flag but not the bonus.
- The notice numbers, the operator and the latest filing date, in the CSV, the lead pack and the report.
- **Major mines are labelled, not rewarded.** Producing mines such as New Afton keep open permits that still include drilling disturbance. Their notices show in `drilling_flag` with `drilling_mine_scale` = Major Mine (and "(major mine)" in the report), but earn no bonus unless you set `drilling_include_major_mines: true`. `drilling_scored_flag` shows the status that actually earned points.

**Read it as a permit signal, not proof of a rig on site.** The layer gives hectares of drilling disturbance and the permit status, not hole counts, depths or schedules. A permit can be approved and never used, administrative amendments are not in the layer, and some notices have no coordinates. Without the layer the flag reads `not assessed` and scores are unchanged.

---

## Drilling indicator on Nevada projects

Nevada has no equivalent of BC's Notice of Work layer. The closest public signal is the **NDEP Bureau of Mining Regulation and Reclamation** (BMRR): a *Reclamation Permit for an Exploration Project* (a "drilling project") means exploration is planned or authorised on that ground.

```powershell
gossan harvest nv_project --source ndep_notices      # reads NDEP's public notice list
gossan ingest-notices nv_project .\notices\          # PDFs you saved yourself (optional)
gossan leads run nv_project
```

- **Coverage is best effort.** NDEP's public list (`ndep.nv.gov/posts/category/land`) is short and rolling, and nothing on the site lists every exploration permit. Each harvest keeps what it saw and the layer accumulates, so run it regularly. For full coverage, subscribe at the bottom of `ndep.nv.gov/posts` ("Get Notices", *Mining Regulation and Reclamation Public Notice*), save the PDFs from the emails, and load them with `gossan ingest-notices`.
- Each notice's **sections, township and range** are turned into section polygons using the BLM national PLSS service, so a notice covers about a square mile per section and counts for **any claim it touches**. That is a coarse match: it means a permitted project area overlaps the claim.
- **Flag and bonus:** a decision posted within the last 18 months is `active` (+10), a proposed action in that window is `pending` (+5), anything older is `lapsed` (+0), and the score is capped at 100. Change `drilling_active_points`, `drilling_pending_points`, `drilling_lapsed_points` and `drilling_window_months` under `parameters: leads:`. Mine permits are kept in the layer but never flag a claim.
- **A permit is not a rig.** NDEP notices give a project and acreage, not hole counts or dates, and a permit can sit unused.

---

## Leads on US projects

`gossan leads run <name>` works on US projects once `gossan target` has run. BLM MLRS has no `ISSUE_DATE` or `GOOD_TO_DATE`, so the recency and expiry scoring used for BC is skipped. Each claim gets up to 100 points:

- **Up to 60** for the share of the claim that overlaps the target polygons (`target/targets.gpkg`).
- **Up to 40** for proximity to MRDS occurrences: full marks with an occurrence on the claim, falling to zero at 2 km.

Change the weights under `parameters: leads:` in `project.yaml` (`overlap_points`, `proximity_points`, `proximity_m`). The CSV has columns `target_overlap_pct`, `nearest_occurrence_m` and `occurrences_inside` instead of the ARIS and date columns. `gossan leads pack` and `gossan report --lead` work too.

**Caveats:** BLM claims are resolved to **quarter-section geocodes, not exact survey boundaries**, so overlap and distances are approximate. **MRDS is a historical inventory** (not systematically updated since 2011), so proximity means "near a documented occurrence", not "near current activity".

---

## Data governance: ARIS "Access Only"

### What it means

The BC Data Catalogue lists the **Assessment Report Indexing System (ARIS)** as **Access Only**, not under the Open Government Licence. Under the Province's copyright policy, Access Only material **may not be reproduced or redistributed without written permission from the Province of British Columbia**.

### What in a Gossan project is ARIS-derived

- The `aris_index` layer.
- Lead scores (`gossan leads run`), which count ARIS reports per tenure.
- The prospectivity model, which uses ARIS report density as an input.
- Scans, georeferenced maps and drillhole tables taken from ARIS report PDFs.

In practice, **treat the whole pitch pack as ARIS-derived.**

### How Gossan marks it

- A red disclaimer banner on page 1.
- A footer on every page: `INTERNAL DRAFT | Contains ARIS data (Access Only): not for distribution`.
- The PDF's metadata subject: `INTERNAL DRAFT - contains ARIS data (Access Only)`.
- An ARIS notice in the *Notices* section of every marked pack.

### When the marking applies (fail-safe)

The marking is applied when ARIS data is recorded in `provenance.jsonl` **or** when anything with `aris` in its name (any case) exists in the project's `raw/`, `rasters/` or `digitize/` folders, or as a layer in `data.gpkg`. The files win over the ledger: if the ledger is missing or incomplete, the pack is still marked. A US-only project with no such files gets a plain draft footer instead. Rename or remove the stray file if the marking is a false alarm; the PDF's provenance record lists what triggered it (`aris_evidence`).

### What you must do

- **Keep the pack internal** to SJS Geospatial Analytics.
- **Before sending it, or anything taken from it, to a client, prospect or other third party:**
  1. Obtain **written permission** from the Province (contact the BC Geological Survey to confirm the terms).
  2. Have a geologist review the content.
  3. Keep a copy of the permission with the project (for example in `<project>/outputs/`).
- **Do not remove or crop the markings** to work around the restriction.
- The same applies to `outputs/leads_ranked.csv`, lead packs (`gossan leads pack`) and the `target/` outputs.

### Other licences in the pack

| Source | Terms |
|---|---|
| BC open data (MINFILE, tenure, RGS) | Open Government Licence – British Columbia: attribution required. |
| NRCan geophysics | Open Government Licence – Canada: attribution required. |
| NASA EMIT / ASTER | CC0 (NASA EOSDIS): free; cite the dataset. |
| EnMAP | DLR EnMAP Data License: no sale, sub-licensing or redistribution. |

The *Notices* section lists the ones that apply to your project automatically.

### Third-party processing

`--extract-scanned-tables` sends page images to **OpenRouter and Anthropic**. Before using it on client-confidential documents (as opposed to public ARIS reports), confirm the client is comfortable with that.

---

## Quick reference

| Task | Command |
|---|---|
| Download ARIS PDFs | `gossan harvest <project> --source aris_index --limit-pdfs N` |
| Digitize all PDFs | `gossan digitize <project>` |
| Read scanned tables with Claude | `gossan digitize <project> --report <aris> --target-pages <pages> --extract-scanned-tables` |
| List pending items | `gossan review list <project>` |
| Approve | `gossan review approve <project> <item_id> --by "<name>" --note "<why>"` |
| Reject | `gossan review reject <project> <item_id> --note "<why>"` |
| Manual georeference | `gossan ingest-points <project> <aris> <page> <file.points>` |
| Pitch pack PDF | `gossan report <project>` |
| Pitch pack for one lead | `gossan report <project> --lead <tenure_number>` |
| Tenure watch (changes and watchlist) | `gossan analyze <project> --steps tenure` |
| Whole job: harvest, model, analyze, report | `gossan run <project> [--aoi <file>] [--class metallic]` |
| Analysis report (PDF and Word) | `gossan report <project> --kind analysis` |

Every command has full help: `gossan <command> --help`.

---

**Previous:** [Tutorial 2: QGIS End-to-End Workflow](02_QGIS_End_to_End_Workflow.md)
