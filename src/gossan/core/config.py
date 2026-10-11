"""Pydantic models for ``project.yaml``."""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_FILE = "project.yaml"
AOI_FILE = "aoi.gpkg"


class Module(StrEnum):
    """Gossan modules that can be enabled for a project."""

    HARVEST = "harvest"
    DIGITIZE = "digitize"
    LEADS = "leads"
    SPECTRAL = "spectral"
    TARGET = "target"
    REPORT = "report"


class GeorefConfig(BaseModel):
    """Settings for georeferencing scanned maps (``gossan digitize``, ``gossan ingest-points``)."""

    model_config = ConfigDict(extra="forbid")

    #: Reject a fit whose root-mean-square GCP error (metres) is above this.
    rms_threshold_m: float = Field(default=15.0, gt=0)
    #: Fewest GCPs the automatic step accepts (4 leaves one redundant, so the RMS means something).
    min_gcps: int = Field(default=4, ge=3)
    #: Datum printed on the maps' grid labels. Old ARIS maps are often NAD27; the labels are
    #: read in this datum's UTM zone (the project's zone) and the result is reprojected.
    datum: Literal["NAD83", "NAD27", "WGS84"] = "NAD83"
    #: Explicit CRS of the grid labels (e.g. "EPSG:26910"); overrides datum + UTM zone.
    gcp_crs: str | None = None
    #: Resolution maps are rendered at for georeferencing.
    dpi: int = Field(default=200, ge=72, le=600)


class SatelliteConfig(BaseModel):
    """Settings for the satellite scene connectors (``gossan harvest --source emit|enmap|aster``)."""

    model_config = ConfigDict(extra="forbid")

    #: STAC date or interval, e.g. "2023-01-01/2024-12-31" or "2022-06-01/..". Unset: all dates
    #: (ASTER is then capped at April 2008, when its SWIR detectors failed).
    datetime: str | None = None
    #: Drop scenes with more cloud than this (percent); scenes with no cloud value are kept.
    max_cloud_percent: float = Field(default=100.0, ge=0, le=100)
    emit_collection: str = "EMITL2ARFL_001"
    enmap_collection: str = "ENMAP_HSI_L2A"
    aster_collection: str = "AST_L1T_004"


class SpectralConfig(BaseModel):
    """Settings for ASTER alteration indices (``gossan spectral``)."""

    model_config = ConfigDict(extra="forbid")

    #: Detector gain the scenes were acquired at. The GeoTIFFs hold raw digital numbers; converting to
    #: radiance needs the gain, which varies by scene and band and is not read from the files. Normal
    #: gain is the usual setting over land; the band ratios shift a few percent if a scene used another.
    gain: Literal["high", "normal", "low1"] = "normal"
    #: Cell size (metres) of the index rasters. ASTER's SWIR bands are 30 m, its VNIR bands 15 m.
    resolution_m: float = Field(default=30.0, gt=0)
    #: Cells whose vegetation index (on at-sensor radiance, which reads lower than on reflectance) is
    #: above this are masked: alteration indices mean nothing under forest, which is most of British Columbia.
    ndvi_max: float = Field(default=0.3, ge=-1, le=1)
    #: Cells whose snow index is above this are masked (snow and ice, and much cloud).
    snow_ndsi: float = Field(default=0.4, ge=-1, le=1)


class GeochemConfig(BaseModel):
    """Settings for stream-sediment catchments (``gossan analyze``, and ``gossan target`` features)."""

    model_config = ConfigDict(extra="forbid")

    #: Cell size (metres) the DEM is resampled to for flow routing. Finer resolves small streams
    #: better but takes longer (50 m: about 20 s for a 3,600 km2 AOI).
    catchment_resolution_m: float = Field(default=50.0, gt=0)
    #: A sample is moved to the highest-flow cell within this distance (samples sit on streams, but
    #: map and DEM streams do not line up exactly).
    snap_m: float = Field(default=200.0, ge=0)
    #: A cell is a stream (a snapping target) when it drains at least this area (km2).
    stream_km2: float = Field(default=0.5, gt=0)
    #: Also remove the effect of catchment size (dilution) from the corrected scores.
    dilution_correction: bool = False
    #: Also remove the effect of catchment lithology (expected background from the bedrock units
    #: each sample drains). Off by default: on the East Kootenay test area both this and the
    #: dilution correction made the pathfinder indices worse at picking out catchments that hold
    #: known metallic occurrences (best-index AUC 0.79 Fe-Mn only, 0.70 with lithology, 0.76 with
    #: dilution), because there the host units themselves carry the mineralization. Turn it on
    #: where barren rocks (black shales, for example) dominate the anomalies. How much of each
    #: element's variation lithology explains is reported either way (``catchment_r2_adjusted``).
    lithology_levelling: bool = False
    #: A unit must make up at least this share of the AOI's catchments to get its own background.
    min_unit_share: float = Field(default=0.02, ge=0, lt=1)


class DrillConfig(BaseModel):
    """Settings for classifying digitized drill holes (``gossan analyze --steps drilling``)."""

    model_config = ConfigDict(extra="forbid")

    #: Screening grades above which an interval counts as mineralized, in each element's canonical
    #: unit (Au, Ag: g/t; the rest: %). They are deliberately far below economic grades: they mean
    #: "mineralization is present here", not "an economic intercept". Elements not listed are ignored.
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "Au": 0.3,
            "Ag": 30.0,
            "Cu": 0.1,
            "Pb": 0.5,
            "Zn": 0.5,
            "Mo": 0.01,
            "Ni": 0.1,
            "Co": 0.02,
            "W": 0.05,
            "Sb": 0.1,
        }
    )
    #: A hole is called barren only if it was sampled over at least this many metres...
    min_metres: float = Field(default=20.0, gt=0)
    #: ...in at least this many intervals...
    min_samples: int = Field(default=5, ge=1)
    #: ...and no interval reached this fraction of any threshold (so a marginal hole is inconclusive).
    barren_fraction: float = Field(default=0.5, gt=0, le=1)
    #: Collar coordinates must land within this distance of the AOI to count as placed; otherwise they
    #: are probably a local mine grid, the wrong zone or datum.
    placement_buffer_m: float = Field(default=3000.0, ge=0)
    #: A target counts as drilled by holes within this distance of it (placed collars)...
    near_m: float = Field(default=500.0, ge=0)
    #: ...or within this distance of a report's location when the hole itself could not be placed.
    report_radius_m: float = Field(default=1500.0, ge=0)


class TenureConfig(BaseModel):
    """Settings for the tenure watchlist (``gossan analyze --steps tenure``)."""

    model_config = ConfigDict(extra="forbid")

    #: Flag a tenure whose good-to date falls within this many days (ground may open up, or the
    #: holder must file work or pay to keep it).
    expiring_days: int = Field(default=180, ge=1)
    #: Flag a tenure issued within this many days (recent staking).
    recent_days: int = Field(default=365, ge=1)
    #: Flag a tenure whose record was updated within this many days (``UPDATE_TIMESTAMP``).
    update_days: int = Field(default=90, ge=1)
    #: Flag a tenure that has changed hands at least this many times.
    churn_transfers: int = Field(default=2, ge=1)


class BrandingConfig(BaseModel):
    """The company's look for Word reports (``gossan.report.branding``). Branding never removes or weakens
    the Access Only marking or the draft banner: it changes appearance, not the disclaimers."""

    model_config = ConfigDict(extra="forbid")

    company: str | None = None
    #: Logo (PNG or JPEG), a path relative to the project folder or absolute. Shown in the page header,
    #: and large on the cover page.
    logo: str | None = None
    #: Colour (``#rrggbb``) of headings, table headers and rules. Ignored by a ``template``'s own styles.
    accent: str = "#2a78d6"
    font: str = "Calibri"
    #: Heading font; None uses ``font``.
    heading_font: str | None = None
    #: Replaces the standard footer text ("DRAFT | Machine-generated: for geologist review") on reports with
    #: no restricted data; on a marked report it is added after the marking.
    footer_text: str | None = None
    #: Contact line for the cover page.
    contact: str | None = None
    #: A ``.docx`` (not ``.dotx``) whose styles, page setup, header and footer the report uses; made with
    #: ``gossan template`` or in Word. Its body text is ignored.
    template: str | None = None
    #: Put a cover page (logo, title, client, date, contact) first.
    cover_page: bool = False

    @field_validator("accent")
    @classmethod
    def _hex_colour(cls, v: str) -> str:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise ValueError("use a colour like #2a78d6")
        return v


class KnowledgeConfig(BaseModel):
    """The knowledge-driven score that sits beside the random forest (``gossan.target.knowledge``)."""

    model_config = ConfigDict(extra="forbid")

    #: Switch the score off to skip it.
    enabled: bool = True
    #: Which deposit model's emphasis to use for the layer weights (starting points from general
    #: exploration geology, not tuned or validated on any area). "generic": any metallic deposit.
    deposit_model: Literal["generic", "sediment_hosted_zn_pb", "polymetallic_vein", "cu_porphyry_skarn", "gold_vein"] = "generic"
    #: Favourable host rocks: a bedrock unit whose name contains any of these words (case-insensitive,
    #: for example "Aldridge" or "limestone") counts as host ground. Empty: the host layer is not used.
    #: This is the geologist's input; choosing units because occurrences sit in them makes the score's
    #: own check circular.
    host_units: list[str] = []
    #: Replace individual layer weights (fault, contact, intrusive, host, geochem, magnetics, alteration);
    #: 0 drops a layer.
    weights: dict[str, float] = {}
    #: Distance (metres) at which proximity to faults, to unit contacts and to intrusive rocks has
    #: faded from fully favourable (on it) to nothing.
    fault_range_m: float = Field(default=3000.0, gt=0)
    contact_range_m: float = Field(default=1500.0, gt=0)
    intrusive_range_m: float = Field(default=5000.0, gt=0)


class TargetConfig(BaseModel):
    """Settings for prospectivity modelling (``gossan target``). See ``gossan.target``."""

    model_config = ConfigDict(extra="forbid")

    #: Grid cell size in metres (``--resolution`` overrides it).
    resolution_m: float = Field(default=100.0, gt=0)
    #: Refuse a grid with more cells than this (bounding box); raise the resolution instead.
    max_cells: int = Field(default=2_000_000, ge=1)
    #: PU-bagging: number of bags, trees per bag, minimum samples per leaf.
    n_bags: int = Field(default=25, ge=2)
    n_trees: int = Field(default=100, ge=5)
    min_samples_leaf: int = Field(default=3, ge=1)
    #: Unlabeled cells drawn per positive in each bag.
    unlabeled_ratio: float = Field(default=1.0, gt=0)
    #: Spatial cross-validation folds, block edge (metres; default: a quarter of the AOI's
    #: shorter side) and the gap left between test blocks and training cells.
    folds: int = Field(default=5, ge=2)
    block_size_m: float | None = Field(default=None, gt=0)
    #: How digitized drill results (``gossan analyze --steps drilling``; collars placed by coordinates
    #: only) enter the model. "evaluate" (default): the model is not changed, but cross-validation also
    #: reports how well occurrences rank above known-barren drill cells (``auc_vs_barren``), the one
    #: place the data hold real negatives. "train": barren holes also become negatives in every bag and
    #: mineralized holes add positives. "off": ignored. "train" has not been validated on real digitized
    #: data (none was available when it was written), so it is opt-in.
    drill_labels: Literal["off", "evaluate", "train"] = "evaluate"
    #: Fewest cells holding a barren hole for them to be used at all.
    min_drill_negatives: int = Field(default=5, ge=1)
    #: Regional magnetic grid (ingested as ``rasters/<name>.tif``; see ``gossan.target.magnetics``).
    #: "derivatives" (default): when such a raster exists, the model gets the field and its derivatives
    #: (local anomaly, vertical derivative, analytic signal, tilt angle, texture) instead of the bare
    #: field. "field": the bare field only (any other raster is used the same way). "off": none.
    magnetics: Literal["derivatives", "field", "off"] = "derivatives"
    #: Name of the magnetic raster; None tries magnetics, nrcan_magnetics, mag, tmi, rtf, rmi.
    magnetic_raster: str | None = None
    #: Upward continuation height (metres) that defines the regional field subtracted for ``mag_local``...
    magnetic_regional_m: float = Field(default=2000.0, gt=0)
    #: ...and the window (metres) for ``mag_texture``.
    magnetic_texture_m: float = Field(default=1000.0, gt=0)
    #: Gap (metres) left between held-out blocks and training cells. Neighbouring cells look alike,
    #: so without a gap the model is tested beside what it trained on and the scores flatter it:
    #: on the East Kootenay test area the effort-matched AUC went 0.63 (no gap), 0.56 (2 km), 0.52
    #: (5 km), 0.45 (10 km). None (default): automatic, 15% of the block edge but at most 2 km.
    #: 0 switches the gap off.
    cv_buffer_m: float | None = Field(default=None, ge=0)
    #: Fewest known occurrences inside the AOI that a model may be trained on.
    min_positives: int = Field(default=10, ge=3)
    seed: int = 42
    #: Cells at or above this percentile of the probability surface become target polygons.
    top_percentile: float = Field(default=95.0, gt=0, lt=100)
    #: Gaussian smoothing (standard deviation, metres) of the score before the percentile is
    #: taken, so isolated high cells merge into the neighbourhoods they sit in. 0 (default): none.
    #: Cross-validation reports the held-out capture with and without it (``capture_top5_smoothed``).
    smooth_m: float = Field(default=0.0, ge=0)
    #: Target polygons smaller than this (hectares) are dropped. One 100 m cell is 1 ha; the default
    #: is about one BC mineral cell claim (16 to 21 ha), the smallest unit anyone can stake.
    min_target_ha: float = Field(default=20.0, ge=0)
    #: Neighbourhood for tenure density and the ARIS report-density kernel (metres).
    density_radius_m: float = Field(default=1000.0, gt=0)
    #: RGS geochemistry interpolation: search radius (metres) and neighbours.
    idw_radius_m: float = Field(default=10_000.0, gt=0)
    idw_neighbours: int = Field(default=8, ge=1)
    geochem_elements: list[str] = [
        "COPPER", "GOLD", "SILVER", "ZINC", "LEAD", "ARSENIC", "MOLYBDENUM", "ANTIMONY",
    ]  # fmt: skip
    #: A feature with more than this fraction of the AOI missing is dropped; the rest are
    #: filled with the feature's median.
    max_missing_fraction: float = Field(default=0.6, ge=0, le=1)
    #: Which known occurrences are the positives, by BC deposit-profile class: "metallic", "industrial",
    #: "specialty" or "all" (see ``gossan.analyze.occurrences``); these need DEPOSIT_TYPE_CODE1.
    #: "auto" (default): metallic where the layer has deposit-type codes and at least ``min_positives``
    #: metallic occurrences fall in the AOI, otherwise all. Metallic-only labels raised the forest's
    #: effort-matched AUC in three of four validated BC areas (+0.04 to +0.06; the fourth was already 96%
    #: metallic), because industrial minerals (gypsum, aggregate) are controlled by other things.
    occurrence_class: Literal["auto", "all", "metallic", "industrial", "specialty"] = "auto"
    #: Exploration effort (ARIS report density, tenure density). "ignore" (default): not used, so
    #: the map is built from geology, geochemistry and terrain alone. "feature": the two layers are
    #: model inputs (the pre-2026-10 behaviour; high plain AUC because occurrences sit where people
    #: explored, so the map largely redraws known camps). "bias": not features, but unlabeled cells
    #: are drawn in proportion to effort (target-group background); on the East Kootenay test
    #: area it scored worst of the three on effort-matched AUC, so it is opt-in.
    effort: Literal["bias", "feature", "ignore"] = "ignore"
    #: Add the label-derived distance to the nearest known occurrence as a feature. It mostly
    #: reproduces known camps, so it is off by default.
    occurrence_distance: bool = False
    #: Add ``unit_favourability``: how enriched each bedrock unit is in (training) occurrences,
    #: rebuilt from the training occurrences only in every cross-validation fold.
    unit_favourability: bool = True
    #: RGS geochemistry features: "indices" (default; deposit-model pathfinder indices and leading
    #: compositional principal components from ``gossan analyze``'s scoring) or "elements" (the
    #: raw ``geochem_elements``).
    geochem_features: Literal["indices", "elements"] = "indices"
    #: How many leading geochemical principal components become features ("indices" mode).
    geochem_components: int = Field(default=3, ge=0, le=6)
    #: How sample scores spread over the grid: "catchment" (default; each cell takes the score of
    #: the sample its water reaches, with IDW where no sample drains it; needs rasters/dem.tif) or
    #: "idw" (inverse-distance weighting only).
    geochem_spatial: Literal["catchment", "idw"] = "catchment"
    #: The knowledge-driven score computed beside the model's (see ``KnowledgeConfig``).
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)


#: Version of the settings layout and defaults. ``gossan new`` writes every setting, defaults
#: included, into ``project.yaml``, so a default changed later in the code would never reach an
#: existing project. When a change matters, bump this and add a migration below: opening an older
#: project then updates the affected values (and says so), instead of silently keeping the old ones.
CONFIG_VERSION = 3


def _v1_to_v2(data: dict[str, Any]) -> list[str]:
    """v2 made the cross-validation gap automatic. A v1 file carries the old default, 0 (no gap
    between test blocks and training cells), which flatters the scores."""
    notes = []
    target = data.get("target")
    if isinstance(target, dict) and target.get("cv_buffer_m") == 0:
        target["cv_buffer_m"] = None
        notes.append(
            "target.cv_buffer_m was 0 (the old default: no gap between cross-validation test blocks and training "
            "cells, which flatters the scores); it is now automatic. Set it back to 0 in project.yaml to keep the old behaviour."
        )
    return notes


def _v2_to_v3(data: dict[str, Any]) -> list[str]:
    """v3 made metallic-only labels the default ("auto"). A v2 file carries the old default, "all"."""
    notes = []
    target = data.get("target")
    if isinstance(target, dict) and target.get("occurrence_class") == "all":
        target["occurrence_class"] = "auto"
        notes.append(
            "target.occurrence_class was 'all' (the old default); it is now 'auto': metallic occurrences where the layer has deposit-type "
            "codes, because metallic-only labels scored higher in three of four validated BC areas (see docs/validation). "
            "Set it back to 'all' in project.yaml to train on every occurrence."
        )
    return notes


#: From version N: a function that updates the settings in place and returns what it changed.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], list[str]]] = {1: _v1_to_v2, 2: _v2_to_v3}


def migrate(data: dict[str, Any]) -> tuple[dict[str, Any], list[str], int]:
    """Bring raw ``project.yaml`` data up to ``CONFIG_VERSION``. Returns (data, notes about changed
    values, the version it started from). A file without ``config_version`` is version 1."""
    start = int(data.get("config_version", 1))
    notes: list[str] = []
    for version in range(start, CONFIG_VERSION):
        notes += MIGRATIONS[version](data)
    data["config_version"] = CONFIG_VERSION
    return data, notes, start


class ProjectConfig(BaseModel):
    """Contents of a project's ``project.yaml``."""

    config_version: int = CONFIG_VERSION
    name: str = Field(min_length=1)
    client: str | None = None
    crs: str = Field(description="Project CRS, e.g. 'EPSG:32610'.")
    utm_zone: int | None = Field(
        default=None, ge=1, le=60, description="UTM zone if the CRS is UTM, else null."
    )
    aoi_path: str = Field(default=AOI_FILE, description="AOI path relative to the project folder.")
    modules: list[Module] = Field(default_factory=lambda: [Module.HARVEST, Module.REPORT])
    #: Which jurisdiction's data sources a plain ``gossan harvest`` uses ("bc" or "us").
    jurisdiction: Literal["bc", "us"] = "bc"
    georef: GeorefConfig = Field(default_factory=GeorefConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    geochem: GeochemConfig = Field(default_factory=GeochemConfig)
    tenure: TenureConfig = Field(default_factory=TenureConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    branding: BrandingConfig = Field(default_factory=BrandingConfig)
    drill: DrillConfig = Field(default_factory=DrillConfig)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_a_folder_name(cls, v: str) -> str:
        if v in {".", ".."} or any(c in v for c in '/\\:*?"<>|'):
            raise ValueError(f"{v!r} is not usable as a folder name")
        return v

    def save(self, project_dir: Path) -> Path:
        """Write ``project.yaml`` into ``project_dir``."""
        path = project_dir / PROJECT_FILE
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, project_dir: Path) -> ProjectConfig:
        """Read and validate ``project.yaml`` from ``project_dir`` (migrated in memory if old)."""
        return cls.read(project_dir)[0]

    @classmethod
    def read(cls, project_dir: Path) -> tuple[ProjectConfig, list[str], int]:
        """Like ``load``, but also returns what migration changed and the file's original version
        (equal to ``CONFIG_VERSION`` when nothing was migrated). Does not write anything."""
        data = yaml.safe_load((project_dir / PROJECT_FILE).read_text(encoding="utf-8"))
        data, notes, start = migrate(data)
        return cls.model_validate(data), notes, start
