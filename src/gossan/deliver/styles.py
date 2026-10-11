"""How each packaged layer looks, shared by the QGIS project and the web map.

Colours follow the report's palette (blue, orange, aqua; neutral greys) so the three deliverables look
alike. Meaning never rests on colour alone: the legend names every layer, and targets are labelled by id.
"""

from __future__ import annotations

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
#: Viridis in five stops: the ramp both viewers use for scores.
RAMP = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")

#: key -> style. ``fill`` None draws an outline only. ``alpha`` is the fill's opacity (0 to 1).
STYLES: dict[str, dict] = {
    "aoi": {"fill": None, "stroke": INK, "width": 0.9, "alpha": 0.0},
    "targets": {"fill": ORANGE, "stroke": "#b8431a", "width": 0.5, "alpha": 0.35},
    "knowledge_targets": {"fill": BLUE, "stroke": "#1c5cab", "width": 0.5, "alpha": 0.25},
    "target_decisions": {"fill": None, "stroke": INK2, "width": 0.4, "alpha": 0.0},
    "mto_tenure": {"fill": None, "stroke": AQUA, "width": 0.4, "alpha": 0.0},
    "bedrock_geology": {"fill": MUTED, "stroke": MUTED, "width": 0.15, "alpha": 0.12},
    "faults": {"stroke": INK, "width": 0.4},
    "minfile": {"fill": INK, "stroke": "#ffffff", "size": 2.2, "alpha": 1.0},
}
DEFAULT_STYLE = {"fill": MUTED, "stroke": INK2, "width": 0.3, "alpha": 0.2, "size": 2.0}


def style_for(key: str) -> dict:
    return {**DEFAULT_STYLE, **STYLES.get(key, {})}


def hex_to_rgb(colour: str) -> tuple[int, int, int]:
    c = colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
