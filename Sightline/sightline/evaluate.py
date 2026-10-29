"""Evaluation helpers (Stage E/H): compare automated candidates with reference moments.

Reference moments can be (a) your own manually noted times, or (b) the dataset's published event
annotations. Both are only as complete as whoever made them, so:
  - "recall" = share of reference moments that some candidate overlaps in time. It is only meaningful
    if the reference list is reasonably complete for the evaluated window.
  - "candidates matching a reference" is NOT precision: an unmatched candidate can be a real event the
    reference list missed. Use human review (confirmed vs rejected) for a false-alarm estimate.
Matching is by time only. It does not check that the same vehicle/pedestrian is involved.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GtEvent:
    gt_id: str
    start_frame: int
    end_frame: int
    cls: int | None = None        # dataset: 0 = vehicle failed to yield, 1 = yielded (annotator judgement)
    source: str = ""


def load_gt_csv(path, fps: float) -> list[GtEvent]:
    """CSV with a header. Columns: time_s OR frame; optional end_time_s OR end_frame; optional note."""
    out = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if row.get("frame"):
                s = int(float(row["frame"]))
            elif row.get("time_s"):
                s = int(round(float(row["time_s"]) * fps))
            else:
                raise ValueError(f"Row {i}: need a 'time_s' or 'frame' column value")
            if row.get("end_frame"):
                e = int(float(row["end_frame"]))
            elif row.get("end_time_s"):
                e = int(round(float(row["end_time_s"]) * fps))
            else:
                e = s
            if e < s:
                raise ValueError(f"Row {i}: end before start")
            out.append(GtEvent(f"manual_{i}", s, e, None, row.get("note", "")))
    if not out:
        raise ValueError("No reference moments in the CSV")
    return out


def _coerce_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_annotation(doc) -> tuple | None:
    """Return (class, group_id) from a dataset annotation JSON, tolerating plain or Labelme-style layouts.

    Unverified against the real files: the dataset README describes keys class / points / group_id.
    """
    if not isinstance(doc, dict):
        return None
    pools = [doc] + [s for s in doc.get("shapes", []) if isinstance(s, dict)]
    for d in pools:
        cls = d.get("class", d.get("label"))
        gid = d.get("group_id")
        if cls is not None or gid is not None:
            return _coerce_int(cls), gid
    return None


def load_gt_annotations_dir(path) -> tuple[list[GtEvent], dict]:
    """Read one video's annotation folder: files named by frame number, grouped by group_id."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Not a folder: {path}")
    groups: dict = {}
    stats = {"files": 0, "parsed": 0, "skipped_name": 0, "skipped_content": 0}
    sample_keys = None
    for f in sorted(path.rglob("*.json")):
        stats["files"] += 1
        if not re.fullmatch(r"\d+", f.stem):
            stats["skipped_name"] += 1
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["skipped_content"] += 1
            continue
        if sample_keys is None and isinstance(doc, dict):
            sample_keys = sorted(doc.keys())
        parsed = parse_annotation(doc)
        if parsed is None:
            stats["skipped_content"] += 1
            continue
        stats["parsed"] += 1
        cls, gid = parsed
        key = ("g", str(gid)) if gid is not None else ("f", f.stem)
        g = groups.setdefault(key, {"frames": [], "cls": None})
        g["frames"].append(int(f.stem))
        if g["cls"] is None:
            g["cls"] = cls
    if not groups:
        raise ValueError(f"No usable annotation files found ({stats}). First file's top-level keys: {sample_keys}. "
                         f"Send me this message so I can adapt the parser.")
    events = [GtEvent(f"gt_{k[1]}", min(g["frames"]), max(g["frames"]), g["cls"], "dataset annotation")
              for k, g in groups.items()]
    events.sort(key=lambda e: e.start_frame)
    return events, stats


def match_events(cands, gts: list[GtEvent], fps: float, tol_s: float = 2.0, window_s: tuple | None = None) -> dict:
    """cands: objects with candidate_key, frame_start, frame_end (candidate Events)."""
    tol = int(round(tol_s * fps))
    if window_s:
        w0, w1 = int(window_s[0] * fps), int(window_s[1] * fps)
        gts = [g for g in gts if g.start_frame <= w1 and g.end_frame >= w0]
        cands = [c for c in cands if c.frame_end >= w0 and c.frame_start <= w1]

    def hit(c, g):
        lo, hi = c.frame_start - tol, c.frame_end + tol
        return lo <= g.start_frame <= hi or lo <= g.end_frame <= hi

    gt_hit = {g.gt_id: [c.candidate_key for c in cands if hit(c, g)] for g in gts}
    cand_hit = {c.candidate_key: [g.gt_id for g in gts if hit(c, g)] for c in cands}
    matched = [g for g in gts if gt_hit[g.gt_id]]
    by_cls = {}
    for cls in sorted({g.cls for g in gts if g.cls is not None}):
        sub = [g for g in gts if g.cls == cls]
        by_cls[str(cls)] = {"total": len(sub), "matched": sum(1 for g in sub if gt_hit[g.gt_id])}
    return {
        "tolerance_s": tol_s,
        "window_s": list(window_s) if window_s else None,
        "reference_moments": len(gts),
        "reference_matched": len(matched),
        "recall_vs_reference": round(len(matched) / len(gts), 3) if gts else None,
        "candidates": len(cands),
        "candidates_matching_a_reference": sum(1 for v in cand_hit.values() if v),
        "unmatched_reference": [{"id": g.gt_id, "time_s": round(g.start_frame / fps, 1), "end_s": round(g.end_frame / fps, 1),
                                 "class": g.cls} for g in gts if not gt_hit[g.gt_id]],
        "unmatched_candidates": [k for k, v in cand_hit.items() if not v],
        "by_reference_class": by_cls,
        "caveats": [
            "Matching is by time only; the same vehicle/pedestrian is not checked.",
            "Recall is only meaningful if the reference list is reasonably complete for this window.",
            "An unmatched candidate is not necessarily a false alarm; use human review for that.",
            "Dataset class 0/1 (failed to yield / yielded) is an annotator judgement about yielding, "
            "not about visibility.",
        ],
    }
