"""Track records, CSV I/O and a reliability summary (Stage C2). Pure logic, no model needed."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrackRow:
    frame: int
    time_s: float
    track_id: int          # -1 = detection without a tracker id
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "auto"   # automated candidate; never a human-confirmed label


FIELDS = [f.name for f in fields(TrackRow)]


def segment_frames(start_s: float, end_s: float, fps: float, frame_count: int, stride: int) -> list[int]:
    """Frame indices from start_s to end_s (inclusive of start) every `stride` frames."""
    if fps <= 0 or frame_count <= 0:
        raise ValueError("fps and frame_count must be positive")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if end_s <= start_s:
        raise ValueError("end must be after start")
    a = max(0, int(round(start_s * fps)))
    b = min(frame_count, int(round(end_s * fps)))
    return list(range(a, b, stride))


def write_tracks_csv(rows: list[TrackRow], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
    return path


def read_tracks_csv(path) -> list[TrackRow]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [TrackRow(int(r["frame"]), float(r["time_s"]), int(r["track_id"]), r["cls"], float(r["conf"]),
                         float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), r.get("source", "auto"))
                for r in csv.DictReader(f)]


def summarize(rows: list[TrackRow], short_obs: int = 5) -> dict:
    """Reliability-oriented summary. Short tracks and class flips are signs of unstable tracking."""
    by_id: dict[int, list[TrackRow]] = defaultdict(list)
    untracked = 0
    for r in rows:
        if r.track_id < 0:
            untracked += 1
        else:
            by_id[r.track_id].append(r)
    majority = {}
    flips = 0
    for tid, rs in by_id.items():
        c = Counter(r.cls for r in rs)
        majority[tid] = c.most_common(1)[0][0]
        if len(c) > 1:
            flips += 1
    per_class = Counter(majority.values())
    short = sum(1 for rs in by_id.values() if len(rs) < short_obs)
    lengths = sorted(len(rs) for rs in by_id.values())
    dup = duplicate_boxes(rows)
    return {
        "detections": len(rows),
        "untracked_detections": untracked,
        "unique_tracks": len(by_id),
        "tracks_per_class": dict(per_class),
        "short_tracks_lt_%d_obs" % short_obs: short,
        "tracks_with_class_flips": flips,
        "track_length_obs_median": lengths[len(lengths) // 2] if lengths else 0,
        "same_frame_duplicate_box_pairs": dup,
        "not_evaluated": ["identity switches and fragmentation (needs ground truth)", "missed detections",
                          "class accuracy", "how many physical objects were present"],
        "note": "'untracked_detections' = 0 only means the tracker gave every box an id; it says nothing about "
                "whether the ids or boxes are right. " + "Track counts are NOT object counts: identity switches and fragmentation inflate them, "
                "and missed detections deflate them.",
    }


def _iou(a, b):
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def duplicate_boxes(rows: list[TrackRow], thr: float = 0.8) -> int:
    """Count same-frame, same-class box pairs with different track ids and IoU >= thr (likely one object twice)."""
    by_frame: dict[int, list[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    n = 0
    for rs in by_frame.values():
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                if a.cls == b.cls and a.track_id != b.track_id and _iou(a, b) >= thr:
                    n += 1
    return n
