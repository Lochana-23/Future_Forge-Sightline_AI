"""Stage C3: draw crosswalk zones on a video frame and save them as JSON.

    python tools\\define_zones.py data\\video_006.avi --frame 1800 --out config\\zones.json

Controls (click the image window first):
  left-click  add a corner to the current zone
  u           undo last corner
  n           finish this zone (you then type its name in the terminal)
  d           delete the last finished zone
  s           save and quit
  q / Esc     quit WITHOUT saving
Zones are in ORIGINAL image pixels. They describe where the crosswalk is in the camera image only.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.fileio import atomic_write_text, backup_file  # noqa: E402
from sightline.paths import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--out", default="config/zones.json")
    ap.add_argument("--max-height", type=int, default=900, help="display height (fits small screens)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(resolve(a.video)))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {a.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        sys.exit(f"Could not read frame {a.frame}")
    h, w = img.shape[:2]
    scale = min(1.0, a.max_height / h)
    zones, cur = [], []

    def render():
        vis = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
        for z in zones:
            pts = [(int(x * scale), int(y * scale)) for x, y in z["polygon"]]
            cv2.polylines(vis, [__import__("numpy").array(pts)], True, (0, 255, 255), 2)
            cv2.putText(vis, z["name"], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for i, (x, y) in enumerate(cur):
            p = (int(x * scale), int(y * scale))
            cv2.circle(vis, p, 4, (0, 0, 255), -1)
            if i:
                cv2.line(vis, (int(cur[i - 1][0] * scale), int(cur[i - 1][1] * scale)), p, (0, 0, 255), 2)
        cv2.imshow("define zones (n=finish zone, u=undo, s=save, q=quit)", vis)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            cur.append((round(x / scale), round(y / scale)))
            render()

    win = "define zones (n=finish zone, u=undo, s=save, q=quit)"
    render()
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("u") and cur:
            cur.pop()
            render()
        elif k == ord("d") and zones:
            print("deleted zone", zones.pop()["name"])
            render()
        elif k == ord("n"):
            if len(cur) < 3:
                print("Need at least 3 corners.")
                continue
            name = input("Name for this zone (e.g. upper_crosswalk): ").strip() or f"zone_{len(zones) + 1}"
            if any(z["name"] == name for z in zones):
                print(f"A zone named '{name}' already exists: names must be unique. Press n and choose another.")
                continue
            zones.append({"name": name, "polygon": [list(p) for p in cur]})
            cur.clear()
            render()
        elif k == ord("s"):
            if not zones:
                print("No finished zones to save (press n to finish a zone first).")
                continue
            out = resolve(a.out)
            bk = backup_file(out, out.parent / "backups", "previous")
            if bk:
                print(f"Existing {out.name} kept as {bk}")
            atomic_write_text(out, json.dumps({"image_size": [w, h], "source_frame": a.frame, "zones": zones}, indent=2))
            print("Saved", out)
            break
        elif k in (ord("q"), 27):
            print("Quit without saving.")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
