"""Figures for the pitch pack: overview and targeting maps, drawn with matplotlib.

Each function returns the PNG path, or None when the data it needs is not in the project (the
report then says so instead of showing an empty figure).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pyogrio
import rasterio

from gossan.core.project import Project

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AOI_COLOUR = "#222222"
OCCURRENCE_COLOUR = "#c0392b"
TARGET_COLOUR = "#ffffff"


def layer_or_none(path: Path, layer: str) -> gpd.GeoDataFrame | None:
    """A layer of a GeoPackage, or None if the file or layer is missing or empty."""
    if not path.exists() or layer not in {str(n) for n in pyogrio.list_layers(path)[:, 0]}:
        return None
    gdf = gpd.read_file(path, layer=layer, engine="pyogrio")
    return gdf if len(gdf) else None


def _finish(fig, ax, dest: Path, crs: str, dpi: int = 130) -> Path:
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=6, labelrotation=20)
    ax.set_xlabel(f"Easting (m), {crs}", fontsize=7)
    ax.set_ylabel("Northing (m)", fontsize=7)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi)
    plt.close(fig)
    return dest


def overview_map(project: Project, dest: Path, label: str = "MINFILE") -> Path:
    """The AOI with tenure, ARIS report locations (if harvested) and the mineral occurrences (``label``)."""
    crs = project.config.crs
    aoi = project.read_aoi()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    handles = []
    tenure = layer_or_none(project.data_gpkg, "mto_tenure")
    if tenure is not None:
        tenure.to_crs(crs).plot(ax=ax, facecolor="#f4d03f55", edgecolor="#b9770e", linewidth=0.6)
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f4d03f55", edgecolor="#b9770e", label=f"Tenure ({len(tenure)})"))
    aris = layer_or_none(project.data_gpkg, "aris_index")
    if aris is not None:
        aris.to_crs(crs).plot(ax=ax, color="#2471a3", markersize=9, marker="s", zorder=3)
        handles.append(plt.Line2D([], [], marker="s", linestyle="", color="#2471a3", markersize=4, label=f"ARIS report location ({len(aris)})"))
    minfile = layer_or_none(project.data_gpkg, "minfile")
    if minfile is not None:
        minfile.to_crs(crs).plot(ax=ax, color=OCCURRENCE_COLOUR, edgecolor="black", markersize=22, zorder=4)
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=OCCURRENCE_COLOUR, markeredgecolor="black", label=f"{label} occurrence ({len(minfile)})"))
    aoi.boundary.plot(ax=ax, color=AOI_COLOUR, linewidth=1.6)
    handles.insert(0, plt.Line2D([], [], color=AOI_COLOUR, linewidth=1.6, label="Area of interest"))
    minx, miny, maxx, maxy = aoi.total_bounds
    pad = 0.04 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    return _finish(fig, ax, dest, crs)


def raster_map(
    project: Project,
    tif: Path,
    dest: Path,
    *,
    title: str,
    cmap: str,
    colourbar_label: str,
    overlay_targets: bool = False,
    overlay_occurrences: bool = False,
) -> Path | None:
    """A single-band raster from ``target/`` drawn with its AOI, optionally with the top-percentile
    target polygons and the known occurrences."""
    if not tif.is_file():
        return None
    crs = project.config.crs
    with rasterio.open(tif) as ds:
        data = ds.read(1, masked=True)
        b = ds.bounds
    if data.mask.all():
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(data, extent=(b.left, b.right, b.bottom, b.top), cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(colourbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    project.read_aoi().boundary.plot(ax=ax, color=AOI_COLOUR, linewidth=1.2)
    handles = [plt.Line2D([], [], color=AOI_COLOUR, linewidth=1.2, label="Area of interest")]
    if overlay_targets:
        targets = layer_or_none(project.root / "target" / "targets.gpkg", "targets_top")
        if targets is not None:
            targets.to_crs(crs).boundary.plot(ax=ax, color=TARGET_COLOUR, linewidth=1.3)
            handles.append(plt.Line2D([], [], color=TARGET_COLOUR, linewidth=1.3, label=f"Top-percentile targets ({len(targets)})"))
    if overlay_occurrences:
        minfile = layer_or_none(project.data_gpkg, "minfile")
        if minfile is not None:
            minfile.to_crs(crs).plot(ax=ax, color=OCCURRENCE_COLOUR, edgecolor="white", markersize=14, linewidth=0.5, zorder=4)
            handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=OCCURRENCE_COLOUR, markeredgecolor="white", label="Known occurrence"))
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_title(title, fontsize=9)
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7, frameon=False)
    legend.set_zorder(10)
    return _finish(fig, ax, dest, crs)


def finite_range(tif: Path) -> tuple[float, float] | None:
    """(min, max) of a raster's valid cells, for captions."""
    if not tif.is_file():
        return None
    with rasterio.open(tif) as ds:
        data = ds.read(1, masked=True)
    if data.mask.all():
        return None
    return float(np.ma.min(data)), float(np.ma.max(data))
