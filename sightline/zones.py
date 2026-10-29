"""Crosswalk zones: user-drawn polygons in ORIGINAL video pixel coordinates (Stage C3)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Zone:
    name: str
    polygon: list  # [(x, y), ...] pixels in the original frame


def validate_zones(zones: list[Zone]):
    names = set()
    for z in zones:
        if not z.name.strip():
            raise ValueError("zone name is empty")
        if z.name in names:
            raise ValueError(f"duplicate zone name: {z.name}")
        names.add(z.name)
        if len(z.polygon) < 3:
            raise ValueError(f"zone '{z.name}' needs at least 3 points")


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray casting. Points exactly on an edge may fall either way (acceptable at pixel scale)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def ground_anchor(x1, y1, x2, y2):
    """Bottom-centre of a box: a rough proxy for where the object touches the ground.
    For an elevated oblique camera this is an approximation, not a measurement."""
    return (x1 + x2) / 2.0, y2


def save_zones(zones: list[Zone], path, frame_size=None) -> Path:
    validate_zones(zones)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"frame_size": frame_size, "note": "Pixel coordinates in the original video frame.",
           "zones": [{"name": z.name, "polygon": [list(map(float, p)) for p in z.polygon]} for z in zones]}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_zones(path) -> list[Zone]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = [Zone(z["name"], [tuple(p) for p in z["polygon"]]) for z in doc["zones"]]
    validate_zones(zones)
    return zones
