"""Stage C1: detection probe on a few frames. Not the full pipeline.

Runs a detector on N evenly spaced frames, saves annotated JPEGs and a CSV, and prints
per-frame counts and timing. Detector output is a CANDIDATE only; confidence is not proof.

Backends:
  yolo (default) - Ultralytics YOLO, CPU. Needs: pip install -r requirements-detect.txt
  motion         - No-download fallback: background subtraction over the ~3 s before each frame.
                   Finds MOVING blobs only; it cannot tell pedestrians from vehicles.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.paths import resolve  # noqa: E402

# COCO class ids we care about
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"moving-object": (0, 255, 255), "person": (0, 0, 255), "bicycle": (0, 200, 255), "car": (0, 200, 0),
          "motorcycle": (255, 200, 0), "bus": (255, 0, 200), "truck": (200, 100, 0)}


def pick_frames(frame_count: int, n: int):
    if frame_count <= 0 or n <= 0:
        return []
    n = min(n, frame_count)
    return sorted({int((i + 0.5) * frame_count / n) for i in range(n)})


def detect_yolo_factory(model_name, imgsz, conf):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install -r requirements-detect.txt\n"
                 "Or use the fallback: --backend motion")
    try:
        model = YOLO(model_name)
    except Exception as e:  # download failure, bad name, etc.
        sys.exit(f"Could not load model '{model_name}': {e}\n"
                 "Check your internet connection, or use the fallback: --backend motion")

    def run(frame, idx=None, cap=None):
        res = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", classes=list(CLASSES), verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append((CLASSES[int(b.cls[0])], float(b.conf[0]), x1, y1, x2, y2))
        return out
    return run


def detect_motion_factory(min_area=150, history_frames=90, stride=3):
    """Fallback: learn the background from frames before idx, then report moving blobs."""
    def run(frame, idx=None, cap=None):
        if cap is None or idx is None:
            return []
        sub = cv2.createBackgroundSubtractorMOG2(history=history_frames, varThreshold=32, detectShadows=True)
        first = max(0, idx - history_frames * stride)
        for j in range(first, idx, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ok, f = cap.read()
            if ok:
                sub.apply(f)
        mask = sub.apply(frame)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadow label (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                out.append(("moving-object", 0.0, float(x), float(y), float(x + w), float(y + h)))
        return out
    return run


def draw(frame, dets):
    img = frame.copy()
    for name, conf, x1, y1, x2, y2 in dets:
        col = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--backend", choices=["yolo", "motion"], default="yolo")
    ap.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name (downloads on first use)")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; larger helps small objects, slower")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default="outputs/probe")
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(resolve(a.video)))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {a.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detect = detect_yolo_factory(a.model, a.imgsz, a.conf) if a.backend == "yolo" else detect_motion_factory()

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"backend={a.backend} model={a.model if a.backend == 'yolo' else '-'} imgsz={a.imgsz} conf>={a.conf}")
    for idx in pick_frames(n_frames, a.frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"frame {idx}: could not decode, skipped")
            continue
        t0 = time.time()
        dets = detect(frame, idx, cap)
        dt = time.time() - t0
        counts = {}
        for d in dets:
            counts[d[0]] = counts.get(d[0], 0) + 1
        print(f"frame {idx:5d} ({idx / fps:6.1f}s): {dt:5.2f}s  {counts or 'no detections'}")
        cv2.imwrite(str(out / f"probe_{idx:06d}.jpg"), draw(frame, dets))
        for name, conf, x1, y1, x2, y2 in dets:
            rows.append({"frame": idx, "time_s": round(idx / fps, 2), "class": name, "confidence": round(conf, 3),
                         "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                         "height_px": round(y2 - y1)})
    with (out / "probe_detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "class", "confidence",
                                          "x1", "y1", "x2", "y2", "height_px"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved annotated frames and probe_detections.csv in {out}")
    print("These are candidates only. Open the JPEGs and check what was missed or wrong.")


if __name__ == "__main__":
    main()
