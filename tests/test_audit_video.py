"""Tests for tools/audit_video.py. Run from the project root: python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_video  # noqa: E402


def make_synthetic_video(path, frames=60, size=(320, 240), fps=30):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 120, np.uint8)
        cv2.rectangle(img, (i * 3, 50), (i * 3 + 40, 100), (255, 255, 255), -1)
        vw.write(img)
    vw.release()


class TestAudit(unittest.TestCase):
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            rep = audit_video.audit_video(Path(d) / "nope.avi", d, False)
        self.assertEqual(rep["error"], "file not found")

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.avi"
            bad.write_bytes(b"this is not a video")
            rep = audit_video.audit_video(bad, d, False)
        self.assertIn("could not open", rep["error"])

    def test_synthetic_video_full_decode(self):
        with tempfile.TemporaryDirectory() as d:
            vid = Path(d) / "synthetic.avi"
            make_synthetic_video(vid, frames=60)
            rep = audit_video.audit_video(vid, Path(d) / "out", True)
        self.assertNotIn("error", rep)
        self.assertEqual(rep["width"], 320)
        self.assertEqual(rep["full_decode"]["frames_read"], 60)
        self.assertTrue(all(s["decoded"] for s in rep["samples"]))
        # The 1200x1100 / 30 FPS expectation should be flagged as mismatching for this clip.
        self.assertTrue(any("Resolution" in w for w in rep["warnings"]))


if __name__ == "__main__":
    unittest.main()
