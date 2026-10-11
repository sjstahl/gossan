"""A QGIS project (``.qgs``) for the package, written as XML, so it needs no QGIS to make.

The layers are the packaged GeoPackage and rasters, referenced by relative path so the folder can be
moved or zipped; each has the package's style. The project is checked by loading it in a real QGIS in
the tests (skipped where none is installed). Anything this writer leaves out (labels, forms, joins) QGIS
fills with its defaults.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from pyproj import CRS

from gossan.deliver.styles import RAMP, hex_to_rgb, style_for

QGIS_VERSION = "3.36.3"


@dataclass
class VectorLayer:
    key: str
    title: str
    geometry: str  # "Polygon", "Line" or "Point"
    path: str  # relative to the project file
    layer: str
    visible: bool = True


@dataclass
class RasterLayer:
    key: str
    title: str
    path: str
    vmin: float
    vmax: float
    opacity: float = 0.7
    visible: bool = True


def _opt(name: str, value: str, kind: str = "QString") -> str:
    return f'<Option name="{name}" value={quoteattr(value)} type="{kind}"/>'


def _rgba(colour: str, alpha: float = 1.0) -> str:
    r, g, b = hex_to_rgb(colour)
    return f"{r},{g},{b},{round(alpha * 255)}"


def _symbol(layer: VectorLayer) -> str:
    s = style_for(layer.key)
    if layer.geometry == "Polygon":
        fill = s["fill"]
        opts = [
            _opt("color", _rgba(fill or "#000000", s["alpha"] if fill else 0.0)),
            _opt("style", "solid" if fill else "no"),
            _opt("outline_color", _rgba(s["stroke"])),
            _opt("outline_style", "solid"),
            _opt("outline_width", str(s["width"])),
            _opt("outline_width_unit", "MM"),
        ]
        kind, cls = "fill", "SimpleFill"
    elif layer.geometry == "Line":
        opts = [_opt("line_color", _rgba(s["stroke"])), _opt("line_style", "solid"), _opt("line_width", str(s["width"])), _opt("line_width_unit", "MM")]
        kind, cls = "line", "SimpleLine"
    else:
        opts = [
            _opt("name", "circle"), _opt("color", _rgba(s.get("fill") or "#000000")), _opt("outline_color", _rgba(s["stroke"])),
            _opt("outline_width", "0.3"), _opt("outline_width_unit", "MM"), _opt("size", str(s.get("size", 2.0))), _opt("size_unit", "MM"),
        ]  # fmt: skip
        kind, cls = "marker", "SimpleMarker"
    return (
        '<renderer-v2 type="singleSymbol" symbollevels="0" forceraster="0" enableorderby="0">'
        f'<symbols><symbol type="{kind}" name="0" alpha="1" clip_to_extent="1" force_rhr="0">'
        '<data_defined_properties><Option type="Map"><Option name="name" value="" type="QString"/><Option name="properties"/>'
        '<Option name="type" value="collection" type="QString"/></Option></data_defined_properties>'
        f'<layer class="{cls}" enabled="1" pass="0" locked="0"><Option type="Map">{"".join(opts)}</Option></layer>'
        "</symbol></symbols></renderer-v2>"
    )


def _srs(crs: CRS) -> str:
    epsg = crs.to_epsg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pyproj warns that a PROJ string loses information; QGIS reads the WKT
        proj4 = crs.to_proj4()
    return (
        f'<spatialrefsys nativeFormat="Wkt"><wkt>{escape(crs.to_wkt())}</wkt><proj4>{escape(proj4 or "")}</proj4>'
        f"<srsid>0</srsid><srid>{epsg or 0}</srid><authid>{escape(f'EPSG:{epsg}' if epsg else '')}</authid>"
        f"<description>{escape(crs.name)}</description><projectionacronym>{'utm' if crs.utm_zone else ''}</projectionacronym>"
        f"<ellipsoidacronym></ellipsoidacronym><geographicflag>{'true' if crs.is_geographic else 'false'}</geographicflag></spatialrefsys>"
    )


def _vector(layer: VectorLayer, srs: str) -> str:
    return (
        f'<maplayer type="vector" geometry="{layer.geometry}" autoRefreshTime="0" autoRefreshEnabled="0" hasScaleBasedVisibilityFlag="0" '
        'minScale="100000000" maxScale="0" styleCategories="AllStyleCategories">'
        f"<id>{layer.key}_gossan</id><datasource>{escape(layer.path)}|layername={escape(layer.layer)}</datasource>"
        f"<layername>{escape(layer.title)}</layername><srs>{srs}</srs><provider encoding=\"UTF-8\">ogr</provider>"
        f"{_symbol(layer)}<layerOpacity>1</layerOpacity></maplayer>"
    )


def _raster(layer: RasterLayer, srs: str) -> str:
    span = layer.vmax - layer.vmin or 1.0
    items = "".join(
        f'<item alpha="255" value="{layer.vmin + span * i / (len(RAMP) - 1):.6g}" label="{layer.vmin + span * i / (len(RAMP) - 1):.3g}" color="{c}"/>'
        for i, c in enumerate(RAMP)
    )
    return (
        '<maplayer type="raster" autoRefreshTime="0" autoRefreshEnabled="0" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0" styleCategories="AllStyleCategories">'
        f"<id>{layer.key}_gossan</id><datasource>{escape(layer.path)}</datasource><layername>{escape(layer.title)}</layername>"
        f"<srs>{srs}</srs><provider>gdal</provider><pipe>"
        f'<rasterrenderer type="singlebandpseudocolor" band="1" opacity="{layer.opacity}" alphaBand="-1" classificationMin="{layer.vmin:.6g}" classificationMax="{layer.vmax:.6g}" nodataColor="">'
        "<rasterTransparency/>"
        f'<rastershader><colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" minimumValue="{layer.vmin:.6g}" maximumValue="{layer.vmax:.6g}" labelPrecision="3">{items}</colorrampshader></rastershader>'
        "</rasterrenderer><brightnesscontrast brightness=\"0\" contrast=\"0\" gamma=\"1\"/><huesaturation colorizeOn=\"0\" saturation=\"0\"/><rasterresampler maxOversampling=\"2\"/>"
        "</pipe></maplayer>"
    )


def write_project(
    path: Path, name: str, crs_text: str, extent: tuple[float, float, float, float], layers: list[VectorLayer | RasterLayer], notice: str = ""
) -> Path:
    """Write the project. ``layers`` run top to bottom as drawn; ``extent`` is (xmin, ymin, xmax, ymax)."""
    crs = CRS.from_user_input(crs_text)
    srs = _srs(crs)
    tree = "".join(
        f'<layer-tree-layer id="{ly.key}_gossan" name={quoteattr(ly.title)} source={quoteattr(ly.path)} '
        f'providerKey="{"gdal" if isinstance(ly, RasterLayer) else "ogr"}" checked="{"Qt::Checked" if ly.visible else "Qt::Unchecked"}" expanded="0"/>'
        for ly in layers
    )
    maps = "".join(_raster(ly, srs) if isinstance(ly, RasterLayer) else _vector(ly, srs) for ly in layers)
    order = "".join(f'<layer id="{ly.key}_gossan"/>' for ly in layers)
    xmin, ymin, xmax, ymax = extent
    text = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
        f'<qgis version="{QGIS_VERSION}" projectname={quoteattr(name)}>'
        f'<homePath path=""/><title>{escape(name)}</title>'
        f"<projectCrs>{srs}</projectCrs>"
        f'<layer-tree-group name="" checked="Qt::Checked" expanded="1">{tree}</layer-tree-group>'
        f'<mapcanvas name="theMapCanvas" annotationsVisible="1"><units>meters</units>'
        f"<extent><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></extent>"
        f"<rotation>0</rotation><destinationsrs>{srs}</destinationsrs></mapcanvas>"
        f"<projectlayers>{maps}</projectlayers>"
        f"<layerorder>{order}</layerorder>"
        '<properties><Paths><Absolute type="bool">false</Absolute></Paths>'
        '<SpatialRefSys><ProjectionsEnabled type="int">1</ProjectionsEnabled></SpatialRefSys>'
        f'<WMSServiceTitle type="QString">{escape(name)}</WMSServiceTitle>'
        f'<WMSServiceAbstract type="QString">{escape(notice)}</WMSServiceAbstract></properties>'
        f"<projectMetadata><title>{escape(name)}</title><abstract>{escape(notice)}</abstract></projectMetadata>"
        "</qgis>"
    )
    path.write_text(text, encoding="utf-8")
    return path
