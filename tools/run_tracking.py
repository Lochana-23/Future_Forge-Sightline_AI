"""Stage C2: detect + track objects over a time window of the video.

Example (60 s window starting at 60 s, every 3rd frame ~ 10 fps):
    python tools\\run_tracking.py data\\video_006.avi --start 60 --end 120 --stride 3

Outputs in --out (default outputs/tracks): tracks.csv, summary.json, preview.avi (optional).
All rows are AUTOMATED CANDIDATES (source=auto). Tracker ids can switch or fragment.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.fileio import atomic_write_text, backup_file  # noqa: E402
from sightline.paths import resolve  # noqa: E402
from sightline.tracks import TrackRow, segment_frames, summarize, write_tracks_csv  # noqa: E402
from sightline.video import VideoError, VideoSource  # noqa: E402

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"person": (0, 0, 255), "bicycle": (0, 200, 255), "car": (0, 200, 0),
          "motorcycle": (255, 200, 0), "bus": (255, 0, 200), "truck": (200, 100, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=60.0, help="window start (s)")
    ap.add_argument("--end", type=float, default=120.0, help="window end (s)")
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/tracks")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
        model = YOLO(a.model)
    except Exception as e:
        sys.exit(f"Could not load Ultralytics/model: {e}\nInstall with: pip install -r requirements-detect.txt")

    try:
        video = VideoSource(resolve(a.video))
    except VideoError as e:
        sys.exit(str(e))
    fps = video.fps
    try:
        frames = segment_frames(a.start, a.end, fps, video.frame_count, a.stride)
    except ValueError as e:
        sys.exit(f"Bad window: {e}")
    if not frames:
        sys.exit("No frames selected (is --start beyond the end of the video?)")

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in (out / "tracks.csv", out / "summary.json"):          # never silently replace an earlier run
        bk = backup_file(old, out / "backups", "previous")
        if bk:
            print(f"Existing {old.name} kept as {bk}")
    writer = None
    rows: list[TrackRow] = []
    t0 = time.time()
    print(f"{len(frames)} frames to process (stride {a.stride}). First frame is slow (model warm-up).")
    interrupted = False
    try:
        for k, (idx, frame) in enumerate(video.iter_frames(frames)):
            if frame is None:
                print(f"frame {idx}: could not decode, skipped")
                continue
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=a.imgsz, conf=a.conf,
                              device="cpu", classes=list(CLASSES), verbose=False)[0]
            vis = frame.copy() if not a.no_preview else None
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                name = CLASSES[int(b.cls[0])]
                tid = int(b.id[0]) if b.id is not None else -1
                conf = float(b.conf[0])
                rows.append(TrackRow(idx, idx / fps, tid, name, conf, x1, y1, x2, y2))
                if vis is not None:
                    col = COLORS.get(name, (255, 255, 255))
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    cv2.putText(vis, f"{name}#{tid}", (int(x1), max(12, int(y1) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if vis is not None:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out / "preview.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                             fps / a.stride, (w, h))
                writer.write(vis)
            if (k + 1) % 20 == 0 or k + 1 == len(frames):
                el = time.time() - t0
                print(f"  {k + 1}/{len(frames)} frames, {el:.0f}s elapsed, ~{el / (k + 1) * (len(frames) - k - 1):.0f}s left")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted: saving what was processed so far.")
    if writer is not None:
        writer.release()
    write_tracks_csv(rows, out / "tracks.csv")
    summary = summarize(rows)
    summary.update(interrupted=interrupted, video=Path(a.video).name, start_s=a.start, end_s=a.end, stride=a.stride,
                   model=a.model, imgsz=a.imgsz, conf=a.conf, fps=fps,
                   seconds_per_frame=round((time.time() - t0) / max(1, len(frames)), 2))
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved tracks.csv, summary.json{'' if a.no_preview else ', preview.avi'} in {out}")


if __name__ == "__main__":
    main()
