# Tutorial 1: Installation and Setup

This tutorial gets Gossan running on your workstation. By the end you will have:

- Gossan installed in its own Python environment, with the `gossan` command working.
- Your API keys (OpenRouter, NASA Earthdata) stored as environment variables.
- The **Gossan: harvest and target** tool available in the QGIS Processing Toolbox.

Allow about 30 minutes. Commands are shown for Windows PowerShell; Linux and macOS equivalents are given where they differ.

---

## 1. Check the prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11 or newer** | Check with `py --version` (Windows) or `python3 --version`. |
| **QGIS 3.x** | Tested on QGIS 3.36.3 (Windows). |
| **Internet access** | The harvest step downloads from DataBC and other public services. |
| **Disk space** | ~2 GB for the environment; several GB more if you install the OCR extra. |

Accounts are only needed for the features that use them:

| Account | Needed for | Required? |
|---|---|---|
| [OpenRouter](https://openrouter.ai) | Reading scanned drillhole/assay tables with Claude vision (`gossan digitize --extract-scanned-tables`) | Only for scanned-table extraction. Paid per use. |
| [NASA Earthdata Login](https://urs.earthdata.nasa.gov) | Downloading EMIT and ASTER satellite scenes | Only for satellite downloads. Free. |
| [EnMAP Instrument Planning Portal](https://planning.enmap.org) | Downloading EnMAP scenes | Only for EnMAP downloads. Free. |

The core QGIS workflow (harvest BC data, build the prospectivity model) needs **no API keys**.

---

## 2. Create a dedicated Python environment

> **Do not install Gossan into QGIS's own Python (OSGeo4W).** The QGIS tool runs Gossan as a separate program, so Gossan's libraries (GeoPandas, rasterio, scikit-learn) never clash with the ones QGIS ships.

Create the environment **outside OneDrive** so thousands of library files are not synced to the cloud:

```powershell
py -3.11 -m venv C:\gossan\venv
C:\gossan\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Your prompt now starts with `(venv)`. Activate the environment this way every time you open a new terminal to use Gossan.

If PowerShell refuses to run `Activate.ps1`, allow local scripts once for your user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Linux / macOS:

```bash
python3.11 -m venv ~/gossan-venv
source ~/gossan-venv/bin/activate
python -m pip install --upgrade pip
```

---

## 3. Install Gossan

From the Gossan repository folder, with the environment active:

```powershell
cd C:\Users\<you>\OneDrive\Desktop\Gossan
pip install -e .
pip install reportlab
```

- `pip install -e .` installs Gossan and its dependencies in *editable* mode, so pulling new code takes effect without reinstalling.
- `reportlab` builds the PDF pitch pack (`gossan report`). It is not yet listed in `pyproject.toml`, so install it explicitly.

### Optional extras

| Extra | Command | Install it if you want to... |
|---|---|---|
| `ocr` | `pip install -e ".[ocr]"` | OCR scanned report pages and let automatic map georeferencing read grid labels (EasyOCR). Pulls in PyTorch: several GB. |
| `dev` | `pip install -e ".[dev]"` | Run the test suite (`pytest`) and linter (`ruff`). |

Without the `ocr` extra, `gossan digitize` still runs; it skips OCR with a warning and most maps will need the manual georeferencing route (see [Tutorial 3](03_Review_and_Reporting.md)).

---

## 4. Verify the installation

```powershell
gossan --help
```

You should see the command list (`new`, `harvest`, `ingest`, `digitize`, `review`, `target`, `report`, `leads`, ...).

Now record **where** the `gossan` program lives. QGIS needs this path:

```powershell
(Get-Command gossan).Source
# e.g. C:\gossan\venv\Scripts\gossan.exe
```

Linux / macOS: `which gossan`.

Optionally, if you installed the `dev` extra, run the tests (no network access needed):

```powershell
pytest
```

---

## 5. Set API keys and environment variables

Gossan reads credentials **only from environment variables**. Never put keys in `project.yaml`, in scripts, or in any file inside the repository.

| Variable | Used by | Value |
|---|---|---|
| `OPENROUTER_API_KEY` | `gossan digitize --extract-scanned-tables` | Your OpenRouter key (`sk-or-...`). |
| `GOSSAN_CLAUDE_MODEL` | Same (optional) | Override the vision model, in OpenRouter `vendor/model` form. Default: `anthropic/claude-opus-5.5`. |
| `EARTHDATA_TOKEN` | `gossan harvest --source emit,aster --limit-scenes N` | A NASA Earthdata Login bearer token. |
| `EOC_USERNAME`, `EOC_PASSWORD` | `gossan harvest --source enmap --limit-scenes N` | Your EnMAP portal account. |
| `GOSSAN_EXECUTABLE` | The QGIS tool (optional) | Full path to `gossan.exe` from step 4. Saves typing it into the tool each time. |

Searching the satellite catalogues needs no credentials; only downloading scenes does.

### Get the keys

**OpenRouter**

1. Sign in at <https://openrouter.ai>.
2. Open **Keys** and click **Create Key**. Copy the key; it is shown only once.
3. Add credits to the account. Every page sent to Claude is billed; Gossan prints the token usage after each run.

**NASA Earthdata**

1. Create a free account at <https://urs.earthdata.nasa.gov>.
2. Sign in, open **Generate Token**, and copy the token.
3. Tokens expire. If downloads start failing with an authorisation error, generate a new one and update the variable.

### Store the variables (Windows)

Set each variable once for your user account:

```powershell
[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-...", "User")
[Environment]::SetEnvironmentVariable("EARTHDATA_TOKEN", "eyJ...", "User")
[Environment]::SetEnvironmentVariable("GOSSAN_EXECUTABLE", "C:\gossan\venv\Scripts\gossan.exe", "User")
```

Then **close and reopen** your terminal and **restart QGIS**. Programs read environment variables only when they start.

Check a variable is set without printing the secret:

```powershell
[bool]$env:OPENROUTER_API_KEY   # True
```

To use a key for the current terminal session only:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

### Store the variables (Linux / macOS)

Add these lines to `~/.bashrc` or `~/.zshrc`, then open a new terminal:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export EARTHDATA_TOKEN="eyJ..."
export GOSSAN_EXECUTABLE="$HOME/gossan-venv/bin/gossan"
```

> Gossan does not read `.env` files. A `.env` file is git-ignored, but its contents are not loaded automatically.

---

## 6. Add `gossan_toolbox.py` to the QGIS Processing Toolbox

The script lives at `qgis/gossan_toolbox.py` in the repository.

1. Start QGIS.
2. Open the Toolbox:
   - **Processing ▸ Toolbox**, or press `Ctrl+Alt+T`.
3. Register the script:
   - In the Toolbox toolbar, click the **Python** icon (Scripts).
   - Choose **Add Script to Toolbox...**.
   - Browse to `qgis\gossan_toolbox.py` in the Gossan repository and click **Open**.
4. Find the tool:
   - Expand **Scripts ▸ Gossan ▸ Gossan: harvest and target**.
   - Or type `gossan` in the Toolbox search box.
5. Double-click it. The dialog opens with the parameters on the left and a help panel on the right. Close it for now; [Tutorial 2](02_QGIS_End_to_End_Workflow.md) walks through a full run.

**Alternative:** copy `gossan_toolbox.py` into one of the folders listed under **Settings ▸ Options ▸ Processing ▸ Scripts ▸ Scripts folder(s)**. QGIS picks it up on the next start.

**After updating the repository:** if `gossan_toolbox.py` changed, add it to the Toolbox again or restart QGIS.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `gossan : The term 'gossan' is not recognized` | The environment is not active. Run `C:\gossan\venv\Scripts\Activate.ps1`. |
| `pip` tries to compile GDAL, rasterio or pyogrio and fails | Your Python version has no prebuilt wheels yet. Recreate the environment with Python 3.11 or 3.12. |
| `ModuleNotFoundError: No module named 'reportlab'` when running `gossan report` | `pip install reportlab` in the Gossan environment. |
| `OPENROUTER_API_KEY is not set` | Set the variable (step 5) and open a new terminal. |
| `EARTHDATA_TOKEN is not set` | Same, for the Earthdata token. Only needed with `--limit-scenes`. |
| QGIS tool says **"Gossan was not found"** | Fill in *Gossan executable* in the tool, or set `GOSSAN_EXECUTABLE` and **restart QGIS**. |
| The tool does not appear in the Toolbox | Use the Toolbox search box; if still missing, repeat step 6 and check the **Log Messages** panel (**View ▸ Panels ▸ Log Messages**, *Processing* tab) for a Python error. |

---

**Next:** [Tutorial 2: QGIS End-to-End Workflow](02_QGIS_End_to_End_Workflow.md)
