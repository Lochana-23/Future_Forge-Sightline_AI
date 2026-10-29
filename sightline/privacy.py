"""Best-effort privacy blur for evidence images. NOT an anonymisation guarantee.

It pixelates the head area (top quarter) of every DETECTED person box and the lower-middle area of every DETECTED
vehicle box (where a number plate usually is). It cannot protect anyone the detector missed, faces seen from
other angles outside those areas, plates on vehicles whose box is wrong, or anything on frames that were never
processed. Check images yourself before publishing any of them, and remember the dataset licence forbids
redistribution of its frames in any case.
"""
from __future__ import annotations

import cv2

PERSON_HEAD_FRACTION = 0.25
PLATE_BAND = (0.55, 1.0)          # vertical share of a vehicle box that is pixelated
PLATE_WIDTH = (0.15, 0.85)        # horizontal share


def _pixelate(img, x1, y1, x2, y2, blocks: int = 5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, min(blocks, x2 - x1)), max(1, min(blocks, y2 - y1))),
                       interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def blur_regions(frame, rows_at_frame):
    """Return a copy of frame with head and plate areas of the detected boxes pixelated."""
    img = frame.copy()
    for r in rows_at_frame:
        bw, bh = r.x2 - r.x1, r.y2 - r.y1
        if r.cls == "person":
            _pixelate(img, r.x1, r.y1, r.x2, r.y1 + PERSON_HEAD_FRACTION * bh)
        else:
            _pixelate(img, r.x1 + PLATE_WIDTH[0] * bw, r.y1 + PLATE_BAND[0] * bh,
                      r.x1 + PLATE_WIDTH[1] * bw, r.y1 + PLATE_BAND[1] * bh)
    return img
