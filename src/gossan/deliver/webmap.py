"""A single-file interactive map (``map.html``) for the package.

No server, no network and no library: the file holds its data (vectors as coordinates in the project
CRS, rasters as coloured PNGs) and a small viewer drawn on a canvas, so it opens from an email
attachment or a USB stick and works offline. There is no basemap, because a basemap needs the network
and its own licence; the AOI, tenure, faults and occurrences give the context, and the QGIS project
holds the rest. The viewer pans, zooms, toggles layers, sets raster opacity, shows coordinates (and
latitude and longitude when the project CRS is UTM) and a scale bar, and a click lists the attributes of
what is under the cursor.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import CRS

from gossan.deliver.items import Item
from gossan.deliver.styles import RAMP, style_for

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

COLORMAP = LinearSegmentedColormap.from_list("ramp", list(RAMP))
SIMPLIFY_M = 8.0
MAX_IMAGE_PX = 1600
MAX_COLUMNS = 24
MAX_TEXT = 160
#: Popup columns for the layers worth curating; every other layer shows all of its attributes.
POPUP = {
    "targets": ["target_id", "rank", "area_ha", "mean_score", "max_score", "n_known_occurrences", "knowledge_pctile", "knowledge_support"],
    "knowledge_targets": ["target_id", "rank", "area_ha", "mean_score", "forest_targets", "overlaps_forest_target"],
}  # fmt: skip
RELATIVE = {"prospectivity", "knowledge", "uncertainty"}


def utm_zone(crs_text: str) -> dict | None:
    """``{"zone": 10, "north": True}`` when the CRS is a UTM zone, else None (no latitude/longitude readout)."""
    zone = CRS.from_user_input(crs_text).utm_zone
    if not zone:
        return None
    return {"zone": int(zone[:-1]), "north": zone[-1] == "N"}


def _clean(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, bool | int):
        return value
    return str(value)[:MAX_TEXT]


def _rings(poly) -> list:
    return [[[round(x, 1), round(y, 1)] for x, y in ring.coords] for ring in (poly.exterior, *poly.interiors)]


def vector_layer(item: Item, gdf: gpd.GeoDataFrame) -> dict | None:
    """The layer as plain lists for the viewer, or None when it has no drawable geometry."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return None
    kind = {"Polygon": "polygon", "MultiPolygon": "polygon", "LineString": "line", "MultiLineString": "line", "Point": "point", "MultiPoint": "point"}
    geom_types = {kind.get(t) for t in gdf.geom_type}
    if len(geom_types) != 1 or None in geom_types:
        return None
    geom = geom_types.pop()
    if geom != "point":
        gdf = gdf.assign(geometry=gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True))
        gdf = gdf[~gdf.geometry.is_empty]
    curated = [c for c in POPUP.get(item.key, []) if c in gdf.columns]
    wanted = (curated + [c for c in gdf.columns if c not in {*curated, "geometry"}])[:MAX_COLUMNS]
    features = []
    for _, row in gdf.iterrows():
        g = row.geometry
        if geom == "polygon":
            parts = [g] if g.geom_type == "Polygon" else list(g.geoms)
            coords = [_rings(p) for p in parts if not p.is_empty]
        elif geom == "line":
            parts = [g] if g.geom_type == "LineString" else list(g.geoms)
            coords = [[[round(x, 1), round(y, 1)] for x, y in p.coords] for p in parts]
        else:
            parts = [g] if g.geom_type == "Point" else list(g.geoms)
            coords = [[round(p.x, 1), round(p.y, 1)] for p in parts]
        props = {c.replace("_", " "): v for c in wanted if (v := _clean(row[c])) is not None}
        features.append({"g": coords, "p": props})
    return {"key": item.key, "title": item.title, "kind": "vector", "geom": geom, "style": style_for(item.key), "features": features, "visible": item.key not in {"knowledge_targets"}, "pick": item.key != "aoi"}


def raster_layer(item: Item) -> dict | None:
    """The raster as a coloured PNG with its bounds, or None when it has no values."""
    with rasterio.open(item.source) as ds:
        step = max(1, math.ceil(max(ds.width, ds.height) / MAX_IMAGE_PX))
        data = ds.read(1, masked=True, out_shape=(ds.height // step, ds.width // step)).astype(float).filled(np.nan)
        b = ds.bounds
    valid = np.isfinite(data)
    if not valid.any():
        return None
    lo, hi = (float(v) for v in np.nanpercentile(data[valid], [2, 98]))
    scaled = np.clip((data - lo) / ((hi - lo) or 1.0), 0, 1)
    rgba = (COLORMAP(np.where(valid, scaled, 0)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    labels = ["low", "high"] if item.key in RELATIVE else [f"{lo:.3g}", f"{hi:.3g}"]
    return {
        "key": item.key, "title": item.title, "kind": "raster", "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bounds": [b.left, b.bottom, b.right, b.top], "labels": labels, "opacity": 0.7, "visible": item.key == "prospectivity",
        "ramp": list(RAMP),
    }  # fmt: skip


def build_map_data(project, vectors: list[tuple[Item, gpd.GeoDataFrame]], rasters: list[Item], notice: str, internal: bool) -> dict:
    """The viewer's data: ``vectors`` as (item, layer) pairs (the AOI as an item keyed ``aoi``), ``rasters`` as items."""
    aoi = project.read_aoi()
    layers: list[dict] = []
    for item in rasters:  # drawn first, under the vectors
        layer = raster_layer(item)
        if layer:
            layers.append(layer)
    order = ["bedrock_geology", "mto_tenure", "target_decisions", "knowledge_targets", "targets", "faults", "minfile", "aoi"]
    for item, gdf in sorted(vectors, key=lambda pair: order.index(pair[0].key) if pair[0].key in order else -1):
        layer = vector_layer(item, gdf)
        if layer:
            layers.append(layer)
    minx, miny, maxx, maxy = aoi.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    return {
        "title": project.config.name,
        "client": project.config.client or "",
        "crs": project.config.crs,
        "utm": utm_zone(project.config.crs),
        "extent": [minx - pad, miny - pad, maxx + pad, maxy + pad],
        "notice": notice,
        "internal": internal,
        "layers": layers,
    }


def write_map(path: Path, data: dict) -> Path:
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = TEMPLATE.replace("__TITLE__", _esc(data["title"])).replace("/*__UTM__*/", UTM_JS).replace("/*__DATA__*/", "const DATA=" + blob + ";")
    path.write_text(html, encoding="utf-8")
    return path


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


#: Inverse transverse Mercator (Krueger series, GRS80), used for the latitude and longitude readout.
UTM_JS = """
function utmToLatLon(E, N, zone, north) {
  const a = 6378137, f = 1 / 298.257222101, k0 = 0.9996, n = f / (2 - f);
  const A = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64);
  const beta = [n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96 - n ** 4 / 360, n ** 2 / 48 + n ** 3 / 15 - 437 * n ** 4 / 1440, 17 * n ** 3 / 480 - 37 * n ** 4 / 840, 4397 * n ** 4 / 161280];
  const delta = [2 * n - 2 * n ** 2 / 3 - 2 * n ** 3, 7 * n ** 2 / 3 - 8 * n ** 3 / 5, 56 * n ** 3 / 15];
  const xi0 = (N - (north ? 0 : 1e7)) / (k0 * A), eta0 = (E - 5e5) / (k0 * A);
  let xi = xi0, eta = eta0;
  for (let j = 1; j <= 4; j++) {
    xi -= beta[j - 1] * Math.sin(2 * j * xi0) * Math.cosh(2 * j * eta0);
    eta -= beta[j - 1] * Math.cos(2 * j * xi0) * Math.sinh(2 * j * eta0);
  }
  let chi = Math.asin(Math.sin(xi) / Math.cosh(eta)), lat = chi;
  for (let j = 1; j <= 3; j++) lat += delta[j - 1] * Math.sin(2 * j * chi);
  const lon0 = (zone * 6 - 183) * Math.PI / 180;
  const lon = lon0 + Math.atan2(Math.sinh(eta), Math.cos(xi));
  return [lat * 180 / Math.PI, lon * 180 / Math.PI];
}
"""

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#dcdad2;--map:#eef1f4;--accent:#2a78d6;--warn:#b8431a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17181a;--panel:#212326;--ink:#f1f0ec;--ink2:#b3b1aa;--line:#34363a;--map:#101214;--accent:#6aa7f0;--warn:#f08a62}}
:root[data-theme=dark]{--bg:#17181a;--panel:#212326;--ink:#f1f0ec;--ink2:#b3b1aa;--line:#34363a;--map:#101214;--accent:#6aa7f0;--warn:#f08a62}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;display:flex;flex-direction:column}
header{padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:17px;margin:0}header .sub{color:var(--ink2);font-size:13px}
.banner{background:var(--warn);color:#fff;padding:5px 16px;font-size:13px;font-weight:600}
main{flex:1;display:flex;min-height:0}
#mapwrap{position:relative;flex:1;min-width:0;background:var(--map)}
canvas{position:absolute;inset:0;touch-action:none;cursor:grab}canvas.drag{cursor:grabbing}
.btns{position:absolute;top:10px;left:10px;display:flex;flex-direction:column;gap:4px}
.btns button{width:34px;height:34px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;font-size:18px;cursor:pointer}
#scale{position:absolute;left:12px;bottom:34px;color:var(--ink);font-size:12px}
#scale div{border:2px solid var(--ink);border-top:none;height:7px;margin-bottom:2px}
#coords{position:absolute;left:12px;bottom:10px;font-size:12px;color:var(--ink2);background:var(--panel);padding:2px 6px;border-radius:4px;border:1px solid var(--line)}
aside{width:320px;max-width:42vw;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:12px 14px}
aside h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:14px 0 6px}aside h2:first-child{margin-top:0}
.layer{display:flex;align-items:center;gap:8px;margin:5px 0}.layer label{flex:1;cursor:pointer}
.sw{width:14px;height:14px;border-radius:3px;border:1px solid var(--ink2);flex:none}
.ramp{height:9px;border-radius:3px;margin:2px 0 0 22px;border:1px solid var(--line)}.rl{display:flex;justify-content:space-between;margin-left:22px;font-size:11px;color:var(--ink2)}
input[type=range]{width:100%;margin:2px 0 0 22px;width:calc(100% - 22px)}
table{border-collapse:collapse;width:100%;font-size:13px}td{padding:3px 6px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{color:var(--ink2);width:42%}
#info .none{color:var(--ink2)}
footer{padding:6px 16px;border-top:1px solid var(--line);background:var(--panel);font-size:12px;color:var(--ink2)}
@media (max-width:760px){main{flex-direction:column}aside{width:auto;max-width:none;max-height:40vh;border-left:none;border-top:1px solid var(--line)}}
@media print{aside,.btns{display:none}}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="sub"></div></header>
<div class="banner" id="banner" hidden></div>
<main><div id="mapwrap"><canvas id="c"></canvas>
<div class="btns"><button id="zin" title="Zoom in" aria-label="Zoom in">+</button><button id="zout" title="Zoom out" aria-label="Zoom out">&minus;</button><button id="fit" title="Whole area" aria-label="Whole area" style="font-size:13px">&#9635;</button></div>
<div id="scale"></div><div id="coords">&nbsp;</div></div>
<aside><h2>Layers</h2><div id="layers"></div><h2>Selected</h2><div id="info"><span class="none">Click a feature to see its attributes.</span></div></aside></main>
<footer id="foot"></footer>
<script>
/*__UTM__*/
/*__DATA__*/
(function () {
  const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
  const view = { cx: 0, cy: 0, s: 1 };
  let W = 0, H = 0, dpr = 1, selected = null;
  const layers = DATA.layers;
  document.getElementById('title').textContent = DATA.title;
  document.getElementById('sub').textContent = [DATA.client && 'Client: ' + DATA.client, 'Coordinates in ' + DATA.crs].filter(Boolean).join('  |  ');
  const foot = document.getElementById('foot');
  foot.textContent = DATA.notice;
  if (DATA.internal) { const b = document.getElementById('banner'); b.hidden = false; b.textContent = 'INTERNAL: this package contains Access Only or otherwise restricted data. Do not send it outside the company.'; }

  layers.forEach(l => {
    if (l.kind === 'raster') { l.img = new Image(); l.img.onload = draw; l.img.src = l.png; }
    else l.features.forEach(f => { f.bb = bbox(l.geom, f.g); });
  });
  function bbox(geom, g) {
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    const add = p => { if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; };
    if (geom === 'polygon') g.forEach(poly => poly[0].forEach(add)); else if (geom === 'line') g.forEach(l => l.forEach(add)); else g.forEach(add);
    return [x0, y0, x1, y1];
  }
  const sx = x => (x - view.cx) / view.s + W / 2, sy = y => H / 2 - (y - view.cy) / view.s;
  const wx = px => (px - W / 2) * view.s + view.cx, wy = py => (H / 2 - py) * view.s + view.cy;

  function resize() {
    const r = canvas.parentElement.getBoundingClientRect(); const first = W === 0;
    W = r.width; H = r.height; dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    if (first) fit(); else draw();
  }
  function fit() {
    const [x0, y0, x1, y1] = DATA.extent; view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
    view.s = Math.max((x1 - x0) / W, (y1 - y0) / H); draw();
  }
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  function rgba(hex, a) { const n = parseInt(hex.slice(1), 16); return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }

  function path(l, f) {
    ctx.beginPath();
    if (l.geom === 'polygon') f.g.forEach(poly => poly.forEach(ring => { ring.forEach((p, i) => i ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))); ctx.closePath(); }));
    else f.g.forEach(line => line.forEach((p, i) => i ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))));
  }
  function draw() {
    if (!W) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const vx0 = wx(0), vx1 = wx(W), vy1 = wy(0), vy0 = wy(H);
    layers.forEach(l => {
      if (!l.visible) return;
      if (l.kind === 'raster') {
        if (!l.img.complete) return; const [x0, y0, x1, y1] = l.bounds;
        ctx.globalAlpha = l.opacity; ctx.imageSmoothingEnabled = false;
        ctx.drawImage(l.img, sx(x0), sy(y1), (x1 - x0) / view.s, (y1 - y0) / view.s); ctx.globalAlpha = 1; return;
      }
      const st = l.style;
      l.features.forEach(f => {
        const b = f.bb; if (b[2] < vx0 || b[0] > vx1 || b[3] < vy0 || b[1] > vy1) return;
        if (l.geom === 'point') {
          const r = (st.size || 2) * 1.6;
          f.g.forEach(p => { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), r, 0, 6.2832); ctx.fillStyle = st.fill || '#000'; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = '#fff'; ctx.stroke(); });
          return;
        }
        path(l, f);
        if (l.geom === 'polygon' && st.fill) { ctx.fillStyle = rgba(st.fill, st.alpha); ctx.fill('evenodd'); }
        ctx.lineWidth = Math.max(1, st.width * 2.2); ctx.strokeStyle = st.stroke; ctx.stroke();
      });
    });
    if (selected) { const [l, f] = selected; ctx.save(); path(l, f); ctx.lineWidth = 3.2; ctx.strokeStyle = css('--accent'); ctx.stroke(); ctx.restore(); }
    scalebar();
  }
  function scalebar() {
    const target = 110 * view.s, mag = Math.pow(10, Math.floor(Math.log10(target)));
    const n = [1, 2, 5, 10].map(k => k * mag).filter(v => v <= target * 1.4).pop() || mag;
    const el = document.getElementById('scale'); el.innerHTML = '<div style="width:' + (n / view.s) + 'px"></div>' + (n >= 1000 ? n / 1000 + ' km' : n + ' m');
  }
  function zoomAt(px, py, k) { const x = wx(px), y = wy(py); view.s = Math.min(Math.max(view.s / k, 0.05), 5e4); view.cx = x - (px - W / 2) * view.s; view.cy = y + (py - H / 2) * view.s; draw(); }

  // --- pointer: drag to pan, wheel and pinch to zoom, click to select -----------------------------------
  const ptrs = new Map(); let moved = 0, last = null;
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.offsetX, e.offsetY]); moved = 0; last = null; canvas.classList.add('drag'); });
  canvas.addEventListener('pointermove', e => {
    readout(e.offsetX, e.offsetY);
    if (!ptrs.has(e.pointerId)) return;
    const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (last) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / last); last = d; moved = 99; return;
    }
    const dx = e.offsetX - prev[0], dy = e.offsetY - prev[1]; moved += Math.abs(dx) + Math.abs(dy);
    view.cx -= dx * view.s; view.cy += dy * view.s; draw();
  });
  const up = e => { if (ptrs.has(e.pointerId) && ptrs.size === 1 && moved < 5) pick(e.offsetX, e.offsetY); ptrs.delete(e.pointerId); last = null; canvas.classList.remove('drag'); };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', e => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  document.getElementById('zin').onclick = () => zoomAt(W / 2, H / 2, 1.6);
  document.getElementById('zout').onclick = () => zoomAt(W / 2, H / 2, 1 / 1.6);
  document.getElementById('fit').onclick = fit;
  function readout(px, py) {
    const x = wx(px), y = wy(py); let t = 'E ' + x.toFixed(0) + '  N ' + y.toFixed(0);
    if (DATA.utm) { const [la, lo] = utmToLatLon(x, y, DATA.utm.zone, DATA.utm.north); t += '   ' + la.toFixed(5) + ', ' + lo.toFixed(5); }
    document.getElementById('coords').textContent = t;
  }

  // --- picking ---------------------------------------------------------------------------------------------------------------
  function inRing(x, y, ring) { let c = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const a = ring[i], b = ring[j]; if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; }
  function segDist(px, py, a, b) { const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy; let t = L ? ((px - a[0]) * dx + (py - a[1]) * dy) / L : 0; t = Math.max(0, Math.min(1, t)); return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy)); }
  function pick(px, py) {
    const x = wx(px), y = wy(py), tol = 7 * view.s; let hit = null;
    for (let i = layers.length - 1; i >= 0 && !hit; i--) {
      const l = layers[i]; if (l.kind !== 'vector' || !l.visible || l.pick === false) continue;
      for (const f of l.features) {
        const b = f.bb; if (x < b[0] - tol || x > b[2] + tol || y < b[1] - tol || y > b[3] + tol) continue;
        let ok = false;
        if (l.geom === 'point') ok = f.g.some(p => Math.hypot(p[0] - x, p[1] - y) < tol);
        else if (l.geom === 'line') ok = f.g.some(line => line.some((p, k) => k && segDist(x, y, line[k - 1], p) < tol));
        else ok = f.g.some(poly => inRing(x, y, poly[0]) && !poly.slice(1).some(h => inRing(x, y, h)));
        if (ok) { hit = [l, f]; break; }
      }
    }
    selected = hit; draw(); info(hit);
  }
  function info(hit) {
    const el = document.getElementById('info'); el.textContent = '';
    if (!hit) { const s = document.createElement('span'); s.className = 'none'; s.textContent = 'Nothing here. Click a feature to see its attributes.'; el.appendChild(s); return; }
    const h = document.createElement('div'); h.style.fontWeight = '600'; h.textContent = hit[0].title; el.appendChild(h);
    const t = document.createElement('table');
    const rows = Object.entries(hit[1].p); if (!rows.length) rows.push(['(no attributes)', '']);
    rows.forEach(([k, v]) => { const tr = t.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = v; });
    el.appendChild(t);
  }

  // --- layer panel (top layer first) --------------------------------------------------------------------------------------
  const panel = document.getElementById('layers');
  layers.slice().reverse().forEach((l, i) => {
    const row = document.createElement('div'); row.className = 'layer';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.id = 'l' + i; cb.checked = l.visible; cb.onchange = () => { l.visible = cb.checked; draw(); };
    const lab = document.createElement('label'); lab.htmlFor = cb.id; lab.textContent = l.title;
    row.append(cb);
    if (l.kind === 'vector') { const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = l.style.fill ? rgba(l.style.fill, Math.max(l.style.alpha, 0.5)) : 'transparent'; sw.style.borderColor = l.style.stroke; row.append(sw); }
    row.append(lab); panel.append(row);
    if (l.kind === 'raster') {
      const bar = document.createElement('div'); bar.className = 'ramp'; bar.style.background = 'linear-gradient(90deg,' + l.ramp.join(',') + ')'; panel.append(bar);
      const lb = document.createElement('div'); lb.className = 'rl'; lb.innerHTML = '<span></span><span></span>'; lb.children[0].textContent = l.labels[0]; lb.children[1].textContent = l.labels[1]; panel.append(lb);
      const r = document.createElement('input'); r.type = 'range'; r.min = 0; r.max = 100; r.value = l.opacity * 100; r.setAttribute('aria-label', 'Opacity of ' + l.title); r.oninput = () => { l.opacity = r.value / 100; draw(); }; panel.append(r);
    }
  });
  window.addEventListener('resize', resize); resize();
  window.__gossan = { view, layers, fit, zoomAt, pick, utmToLatLon, state: () => ({ W, H, selected: selected && selected[1].p }) };
})();
</script></body></html>
"""
