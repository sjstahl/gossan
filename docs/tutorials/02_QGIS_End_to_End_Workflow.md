# Tutorial 2: QGIS End-to-End Workflow

This tutorial takes you from an empty QGIS map to a ranked prospectivity map in one run. You will:

1. Draw an Area of Interest (AOI) polygon.
2. Configure and run the **Gossan: harvest and target** tool.
3. Inspect `prospectivity.tif` and the ranked target polygons from `targets.gpkg` on the map canvas.

**Before you start:** complete [Tutorial 1](01_Installation_and_Setup.md). This workflow needs internet access but **no API keys**.

---

## What the tool does

One click runs three Gossan commands in sequence, then loads the results:

| Step | Command run for you | Result |
|---|---|---|
| 1 | `gossan new <name> --aoi <your AOI>` | Project folder `<base>/<name>/`, AOI reprojected to the UTM zone of its centroid. |
| 2 | `gossan harvest <name>` | MINFILE occurrences, MTO mineral tenure, ARIS report index and RGS geochemistry downloaded into `data.gpkg`. |
| 3 | `gossan target <name> --resolution <m>` | PU-bagging random forest model, spatial cross-validation, outputs in `<name>/target/`. |
| 4 | *(QGIS)* | Prospectivity raster and target polygons added to your map, styled. |

Gossan runs as a separate program in its own Python environment. QGIS only starts it and shows its log.

---

## 1. Plan the AOI

The model learns from **known MINFILE occurrences inside the AOI**. It needs **at least 10**, so:

- Use a **regional** AOI (a belt, a district, a map sheet), not a single claim block.
- Include the geological setting you are targeting plus enough known showings to learn from.

The grid is limited to **2,000,000 cells** over the AOI's bounding box. This sets the largest AOI for each resolution:

| Grid resolution | Largest bounding box (approx.) | Typical use |
|---|---|---|
| 250 m | 125,000 km² | Quick regional screen |
| 100 m (default) | 20,000 km² | District-scale targeting |
| 50 m | 5,000 km² | Camp-scale follow-up |

A finer grid than your input data supports (for example, much finer than the RGS stream-sediment sample spacing) adds run time, not information.

---

## 2. Draw the AOI polygon

Any CRS is fine; Gossan reprojects to UTM. The layer **must** have a CRS defined.

1. Add a basemap for orientation:
   - In the **Browser** panel, expand **XYZ Tiles** and double-click **OpenStreetMap**.
2. Create a polygon layer:
   - **Layer ▸ Create Layer ▸ New GeoPackage Layer...**
   - **Database:** click **...** and save as, for example, `D:\GossanProjects\aoi\skeena_aoi.gpkg`.
   - **Table name:** `aoi`.
   - **Geometry type:** **Polygon**.
   - **CRS:** choose a projected or geographic CRS, for example *EPSG:3005 NAD83 / BC Albers*.
   - Click **OK**.
3. Draw the polygon:
   - Select the new layer in the **Layers** panel.
   - Click **Toggle Editing** (pencil icon).
   - Click **Add Polygon Feature** (`Ctrl+.`).
   - Left-click to place each vertex; **right-click** to finish.
   - Accept the attribute form (an `fid` is enough).
4. Save:
   - Click **Save Layer Edits**, then **Toggle Editing** again to stop editing.

**Already have a boundary?** Any polygon layer works (a shapefile of a district, a claim group outline, a tenure selection). To use only some of its polygons, select them first with **Select Features** and tick **Selected features only** in step 4 below.

> Keep project data out of the Git repository. Use a separate data folder such as `D:\GossanProjects`, or the repository's `projects\` folder, which is git-ignored.

---

## 3. Open the Gossan tool

- Open the Toolbox: **Processing ▸ Toolbox** (`Ctrl+Alt+T`).
- Expand **Scripts ▸ Gossan** and double-click **Gossan: harvest and target**.

---

## 4. Configure the parameters

| Parameter | What to enter | Example |
|---|---|---|
| **Area of interest (polygon layer)** | Your AOI layer. Tick **Selected features only** to use just the selected polygons. | `aoi` |
| **Project name (becomes the folder name)** | A short name without `/ \ : * ? " < > \|`. Becomes `<base>/<name>/`. | `skeena_2026` |
| **Base directory** | Existing folder in which the project folder is created. | `D:\GossanProjects` |
| **Gossan executable** | Path to `gossan.exe` in the Gossan environment. Leave blank if `GOSSAN_EXECUTABLE` is set or `gossan` is on `PATH`. | `C:\gossan\venv\Scripts\gossan.exe` |
| **Grid resolution (metres)** | Model cell size. Coarser is faster. See the table in section 1. | `100` |
| **Reuse the project folder if it already exists** | Tick to re-run harvest and target in an existing project. **The AOI layer is then ignored.** Leave unticked for a new project. | unticked |

The tool checks the inputs before it starts. It refuses to run if the name is invalid, the base directory does not exist, the project folder already exists (and *Reuse* is unticked), or Gossan cannot be found.

---

## 5. Run the pipeline

1. Click **Run**.
2. Watch the **Log** tab. Each command is printed, followed by its output:

   ```text
   --- gossan new ---
   Created project skeena_2026 (CRS EPSG:32609)
   --- gossan harvest ---
   minfile: <n> ...
   mto_tenure: <n> ...
   aris_index: <n> ...
   rgs: <n> ...
   --- gossan target ---
   Grid: <w> x <h> cells (...) at 100 m in EPSG:32609; ...
   Known occurrences used as positives: <n>
   Features (<n>): ...
   Spatial CV (5 fold(s)): AUC <x> +/- <y>, top-5% capture <x>% +/- <y>% (random: 5%)
   Targets: <n> polygon(s) at or above the 95th percentile (score <x>)
   ```

3. Expect the harvest to take **several minutes**. The progress bar moves to about 60% when it finishes.

What to know while it runs:

- **A harvest source fails:** the tool reports it in red and **continues** with what was downloaded. `gossan target` then stops with a clear message only if the missing data matters.
- **Cancel** stops the running Gossan command immediately.
- **A layer listed as `skipped`** under *Features* means that input was missing or too sparse in your AOI. The model is built from the rest.

---

## 6. Inspect the results on the map

When the run finishes, two layers are added to the **Layers** panel, already styled:

| Layer | Source file | Styling |
|---|---|---|
| **Gossan targets (top percentile)** | `<project>/target/targets.gpkg`, layer `targets_top` | Hollow polygons, white outline, drawn on top. |
| **Gossan prospectivity** | `<project>/target/prospectivity.tif` | Viridis ramp: dark purple = low, yellow = high. |

Work through them:

- **Turn on the targets.** Each polygon is a cluster of cells at or above the 95th percentile of the score.
- **Query a target:** pick the **Identify Features** tool (`Ctrl+Shift+I`) and click a polygon. Open the full list with right-click ▸ **Open Attribute Table**.
- **Add the evidence layers** to see what drives each target:
  - **Layer ▸ Add Layer ▸ Add Vector Layer...**, choose `<project>/data.gpkg`, and add `minfile`, `mto_tenure`, `aris_index` and `rgs`.
  - Or drag `data.gpkg` from the **Browser** panel onto the map and pick the layers.
- **Add the uncertainty raster:** drag `<project>/target/uncertainty.tif` onto the map. High values mean the bagged models disagree; treat targets there with caution.
- **Save the QGIS project** inside the project folder (**Project ▸ Save As...**, for example `<project>/skeena_2026.qgz`) so the layout is kept with the data.

### Other outputs in `<project>/target/`

| File | What it holds |
|---|---|
| `prospectivity.tif` | Relative prospectivity score per cell. |
| `uncertainty.tif` | Spread of the score across the bagged models. |
| `targets.gpkg` | Target polygons (`targets_top`). |
| `feature_importance.csv` | How much each input layer contributed. Open in Excel or add to QGIS as a table. |
| `cv_report.json` | Spatial cross-validation results per fold. |
| `feature_stack.tif`, `grid.json` | The model inputs and grid definition, for reproducibility. |

Every command and dataset is also recorded in `<project>/provenance.jsonl`.

---

## 7. Read the result responsibly

- **It is a relative ranking, not a probability of a deposit.** A yellow cell resembles the known occurrences more than a purple one does; nothing more.
- **Check the cross-validation numbers.** *Top-5% capture* is the share of held-out occurrences that fell in the top 5% of cells. Compare it with the 5% a random map would score. An AUC near 0.5 means the model does no better than chance.
- **Exploration bias.** Tenure density and ARIS report density are inputs, so areas that have already been explored score higher partly *because* they were explored. Look for targets that are not simply clusters of old claims.
- **ARIS data is "Access Only".** The outputs are built from it, so keep them internal. See [Tutorial 3](03_Review_and_Reporting.md#data-governance-aris-access-only).

---

## US projects (Nevada, Arizona)

When the AOI lies in the United States, `gossan new` sets `jurisdiction: us` and picks the UTM zone; harvest then downloads **USGS MRDS** occurrences and **BLM MLRS** mining claims instead of the BC sources. Read the results with these limits in mind:

- **BLM claims are quarter-section geocodes, not exact survey boundaries.** Each claim polygon is the PLSS quarter section(s) the claim is recorded against. It can be much larger or smaller than the staked ground, and its edges are only approximate. Do not use claim overlap or distances for legal or staking decisions.
- **MRDS is a historical inventory.** USGS has not systematically updated it since 2011, and coordinates come from many sources of varying quality. Treat MRDS points as "documented past occurrences", not current activity or complete coverage. They are the positives the model learns from, so recent discoveries are missing.
- **BLM MLRS has no issue date, good-to date or claimant.** Those fields are empty in `mto_tenure`.
- **Lead scoring works differently.** After `gossan target`, `gossan leads run <name>` scores each claim on two spatial signals only: how much of it lies inside the target polygons (`target/targets.gpkg`), and how close it is to an MRDS occurrence. Recency and expiry are skipped. See [Tutorial 3](03_Review_and_Reporting.md#leads-on-us-projects).

---

## 8. Re-run or adjust

**Change the resolution on an existing project** (no re-drawing):

- Open the tool, enter the same **Project name** and **Base directory**.
- Tick **Reuse the project folder if it already exists**.
- Set the new **Grid resolution** and click **Run**.

**Equivalent command line**, from the base directory, with the Gossan environment active:

```powershell
cd D:\GossanProjects
gossan new skeena_2026 --aoi D:\GossanProjects\aoi\skeena_aoi.gpkg --client "Example Resources Ltd."
gossan harvest skeena_2026
gossan target skeena_2026 --resolution 100
```

Model settings (number of bags, CV folds, the target percentile, geochemistry elements) are in `<project>/project.yaml` under `target:`. Command-line options such as `--n-bags`, `--folds` and `--seed` override them for one run; see `gossan target --help`.

> When run from the command line (or via `processing.run(...)` in the QGIS Python console), the layers are **not** loaded automatically. Add `target/prospectivity.tif` and `target/targets.gpkg` by hand.

---

## 9. Troubleshooting

| Message or symptom | Fix |
|---|---|
| **"Gossan was not found"** | Set **Gossan executable** to the full path of `gossan.exe`, or set `GOSSAN_EXECUTABLE` and restart QGIS. |
| **"... already exists. Pick another name, or tick 'Reuse the project folder'."** | Choose a new name, or tick *Reuse* to update the existing project. |
| **`gossan target failed ... at least 10 are needed`** | Too few MINFILE occurrences in the AOI. Enlarge the AOI. Several occurrences in the same grid cell count as one. |
| **`... would have N cells (limit 2,000,000); use a coarser --resolution`** | Use a coarser **Grid resolution** or a smaller AOI. |
| **A DLL error or `ImportError` in the log** | Gossan picked up QGIS's libraries. The tool removes the usual QGIS variables; if you start QGIS with unusual ones, start it from a clean shell instead. |
| **`gossan harvest reported a problem`** | One source failed (often a temporary DataBC outage). Re-run with *Reuse* ticked, or load the data yourself with `gossan ingest` (see `gossan ingest --help`). |
| **Nothing loaded on the map** | Read the end of the log. If `gossan target` printed an error, no outputs were written. |

---

**Previous:** [Tutorial 1: Installation and Setup](01_Installation_and_Setup.md) · **Next:** [Tutorial 3: Review and Reporting](03_Review_and_Reporting.md)
