"""Small pure helpers used by the Streamlit app (kept here so they can be unit-tested)."""
from __future__ import annotations

import re


def filter_events(events, store, status: str = "all") -> list:
    """Candidates sorted by time, optionally limited to one review status ('unreviewed' includes no-review)."""
    evs = sorted(events, key=lambda e: (e.time_s, e.event_id))
    if status == "all":
        return evs
    return [e for e in evs if store.get(e.candidate_key).status == status]


def clamp_pos(pos, n: int) -> int:
    if n <= 0:
        return 0
    try:
        pos = int(pos)
    except (TypeError, ValueError):
        pos = 0
    return max(0, min(pos, n - 1))


def parse_points(text: str, min_points: int = 3) -> list[list[float]]:
    """'x,y; x,y; x,y' (or one pair per line) -> [[x, y], ...]. Raises ValueError with a clear message."""
    parts = [p.strip() for p in re.split(r"[;\n]", text or "") if p.strip()]
    pts = []
    for p in parts:
        nums = [q for q in re.split(r"[,\s]+", p) if q]
        if len(nums) != 2:
            raise ValueError(f"'{p}' is not an 'x,y' pair")
        try:
            pts.append([float(nums[0]), float(nums[1])])
        except ValueError:
            raise ValueError(f"'{p}' contains a non-number") from None
    if len(pts) < min_points:
        raise ValueError(f"need at least {min_points} points, got {len(pts)}")
    return pts


def parse_box(text: str):
    """'x1,y1,x2,y2' -> (x1, y1, x2, y2) floats, or None if empty/malformed."""
    try:
        v = [float(x) for x in str(text).split(",")]
        return tuple(v) if len(v) == 4 else None
    except ValueError:
        return None


def foot_of_box(text: str):
    """Bottom-centre of an 'x1,y1,x2,y2' box string, or None."""
    b = parse_box(text)
    return None if b is None else ((b[0] + b[2]) / 2, b[3])
