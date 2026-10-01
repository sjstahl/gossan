# Gossan in the QGIS Processing Toolbox

`gossan_toolbox.py` adds one algorithm, **Gossan > Gossan: harvest and target**. Give it an area of
interest and it runs `gossan new`, `gossan harvest` and `gossan target`, then loads the
prospectivity raster and the top-percentile target polygons into your QGIS project.

Gossan is **not** imported into QGIS. The script starts the `gossan` command as a separate
process, so Gossan's packages (GeoPandas, rasterio, scikit-learn and others) stay in their own
Python environment and cannot clash with QGIS's. The child process is started without QGIS's
`PYTHONHOME`, `PYTHONPATH`, `GDAL_DATA`, `PROJ_LIB` and similar settings, and without QGIS and
OSGeo4W folders on its `PATH`.

Written for QGIS 3.x; tested on QGIS 3.36.3 (Windows).

## Before you start

1. **Install Gossan in its own environment**, outside QGIS (for example a virtual environment):

   ```
   pip install -e path\to\Gossan
   ```

2. **Find the `gossan` program** in that environment: `...\Scripts\gossan.exe` on Windows,
   `.../bin/gossan` elsewhere. The algorithm needs this path (see *Parameters*).
3. You need internet access for the harvest step (BC data services) and an AOI that holds at least
   **10 known MINFILE occurrences**. A small claim polygon will not do: use a regional AOI.

## Add the script to the Toolbox

1. Open **Processing > Toolbox** (`Ctrl+Alt+T`).
2. In the Toolbox toolbar click the **Python** (script) button and choose
   **Add Script to Toolbox...**.
3. Select `qgis/gossan_toolbox.py` from this repository and click **Open**.
4. Find the algorithm under **Scripts > Gossan > Gossan: harvest and target** (use the search box if
   it is not visible). If you edit the script later, add it again or restart QGIS.

You can also copy the file into your profile's scripts folder
(**Processing > Options > Scripts > Scripts folder(s)**) and it is picked up on the next start.

## Parameters

| Parameter | What to enter |
|---|---|
| Area of interest | A polygon layer. Selected features are used if some are selected. |
| Project name | Becomes the project folder name (no `/ \ : * ? " < > \|`). |
| Base directory | Folder in which `<project name>/` is created. |
| Gossan executable | Path to `gossan` / `gossan.exe`. Leave blank to use the `GOSSAN_EXECUTABLE` environment variable, then `PATH`. |
| Grid resolution | Cell size in metres for the model (default 100). Coarser is faster. |
| Reuse the project folder | Tick to re-run `harvest` and `target` in a folder that already exists. The AOI layer is then ignored. Without it an existing folder is refused. |

## What happens

1. The AOI layer is exported to a temporary GeoPackage and `gossan new <name> --aoi ...` creates
   `<base>/<name>/` (it picks the UTM zone and reprojects the AOI).
2. `gossan harvest <name>` downloads MINFILE, MTO tenure, the ARIS index and RGS into
   `data.gpkg`. This can take several minutes. If one source fails the run continues with what was
   downloaded; the problem appears in the log.
3. `gossan target <name> --resolution N` builds the grid, runs spatial cross-validation and writes
   `<name>/target/`.
4. Two layers are added to the current project and drawn with symbology already applied:
   - **Gossan prospectivity**: `target/prospectivity.tif` with a viridis pseudocolour ramp from the
     layer's lowest to highest value.
   - **Gossan targets (top percentile)**: layer `targets_top` of `target/targets.gpkg`, hollow
     polygons with a white outline, drawn above the raster. Skipped if no polygons were produced.

The log shows every command and its output. **Cancel** stops the running command.
The other files (uncertainty raster, feature importance, cross-validation report, grid description)
are in the project's `target/` folder; the full run is recorded in `provenance.jsonl`.

Layers are added when the algorithm finishes, from the Processing dialog. If you run it from the
Python console with `processing.run(...)`, QGIS does not run that final step, so the layers are
not loaded; add them from the project folder by hand.

## Reading the result

The raster is a **relative ranking** of how much each cell resembles known occurrences, not a
probability of a deposit. It leans on tenure and report density, which partly reflect where people
have already explored. The ARIS index is **Access Only**, so keep the outputs internal. See
`gossan report` for the full write-up with cross-validation tables.

## Troubleshooting

- **"Gossan was not found"**: set *Gossan executable* to the full path of `gossan.exe`, or set the
  `GOSSAN_EXECUTABLE` environment variable and restart QGIS.
- **"already exists"**: pick another project name, or tick *Reuse the project folder*.
- **`gossan target failed ... at least 10 are needed`**: the AOI holds too few MINFILE occurrences.
  Use a larger area, or a coarser resolution.
- **A DLL or `ImportError` message from `gossan`**: it means the child process picked up QGIS's
  libraries. The script removes the usual variables; if you start QGIS with unusual ones, unset them
  or start QGIS from a clean shell.
- **Nothing loaded**: open the log. If `gossan target` printed an error, nothing was written, and
  the algorithm stops there.
