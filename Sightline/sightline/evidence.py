"""Evidence images: annotated frames and multi-frame contact sheets (Stage E).

A contact sheet is used instead of a video clip because browsers cannot reliably play the
MPEG-1 .avi source and H.264 encoders are not guaranteed on a student PC.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def parse_ids(s) -> list[int]:
    """'1;2;7' -> [1, 2, 7]. Accepts a list too. Ignores blanks."""
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(";") if str(x).strip() != ""]


def draw_event_boxes(frame, zone_poly, vehicle_ids, ped_ids, rows_at_frame):
    """Return a copy of frame with the zone outline and the boxes of the event's tracks."""
    img = frame.copy()
    if zone_poly:
        cv2.polylines(img, [np.array(zone_poly, dtype=np.int32)], True, (0, 255, 255), 2)
    for r in rows_at_frame:
        if r.cls == "person" and r.track_id in ped_ids:
            col = (0, 0, 255)
        elif r.cls != "person" and r.track_id in vehicle_ids:
            col = (0, 200, 0)
        else:
            continue
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
        cv2.putText(img, f"{r.cls}#{r.track_id}", (int(r.x1), max(12, int(r.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def choose_sheet_frames(processed_frames, start, end, evidence, fps, n=6, pad_s=1.0) -> list[int]:
    """Pick up to n processed frames spanning [start-pad, end+pad], always including the evidence frame."""
    lo, hi = start - int(pad_s * fps), end + int(pad_s * fps)
    cand = sorted(f for f in set(processed_frames) if lo <= f <= hi)
    if not cand:
        return [evidence]
    if len(cand) > n:
        step = (len(cand) - 1) / (n - 1)
        cand = sorted({cand[round(i * step)] for i in range(n)})
    if evidence not in cand:
        cand = sorted(set(cand[: n - 1] + [evidence])) if len(cand) >= n else sorted(cand + [evidence])
    return cand


def make_sheet(images: list, labels: list[str], cols: int = 3, thumb_w: int = 420):
    """Tile images into one grid with a caption bar on each tile."""
    if not images:
        raise ValueError("no images")
    tiles = []
    for img, lab in zip(images, labels):
        h, w = img.shape[:2]
        th = int(round(h * thumb_w / w))
        t = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)
        bar = np.zeros((22, thumb_w, 3), np.uint8)
        cv2.putText(bar, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    th = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((th - t.shape[0], t.shape[1], 3), np.uint8)]) if t.shape[0] < th else t
             for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def build_event_sheet(vid, rows_by_frame: dict, zone_poly, ev, out_path, n: int = 6) -> Path | None:
    """Contact sheet for one candidate Event (needs a VideoSource, tracks grouped by frame, the zone polygon)."""
    fps = vid.fps
    start = int(ev.frame_start) if int(ev.frame_start) >= 0 else int(ev.frame)
    end = int(ev.frame_end) if int(ev.frame_end) >= 0 else int(ev.frame)
    frames = choose_sheet_frames(rows_by_frame.keys(), start, end, int(ev.frame), fps, n)
    vids, peds = parse_ids(ev.vehicle_tracks), parse_ids(ev.pedestrian_tracks)
    images, labels = [], []
    for f, frame in vid.iter_frames(frames):
        if frame is None:
            continue
        images.append(draw_event_boxes(frame, zone_poly, vids, peds, rows_by_frame.get(f, [])))
        labels.append(f"frame {f} | {f / fps:.1f}s" + ("  <- evidence" if f == int(ev.frame) else ""))
    if not images:
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), make_sheet(images, labels)):
        return None
    return out

