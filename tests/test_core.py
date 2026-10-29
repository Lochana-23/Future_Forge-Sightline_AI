"""Core-logic tests for Stage B. Run: python -m unittest discover -s tests -v"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.events import EventLog  # noqa: E402
from sightline.video import VideoError, VideoSource  # noqa: E402


def make_video(path, frames=60, size=(320, 240), fps=30):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 100, np.uint8)
        cv2.rectangle(img, (i * 3, 50), (i * 3 + 40, 100), (255, 255, 255), -1)
        vw.write(img)
    vw.release()


class TestVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.vid_path = self.dir / "t.avi"
        make_video(self.vid_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_and_corrupt(self):
        with self.assertRaises(VideoError):
            VideoSource(self.dir / "nope.avi")
        bad = self.dir / "bad.avi"
        bad.write_bytes(b"not a video")
        with self.assertRaises(VideoError):
            VideoSource(bad)

    def test_metadata_and_clamp(self):
        v = VideoSource(self.vid_path)
        self.assertEqual((v.width, v.height, v.frame_count), (320, 240, 60))
        self.assertEqual(v.clamp(-5), 0)
        self.assertEqual(v.clamp(999), 59)
        self.assertEqual(v.frame_of(1.0), 30)
        self.assertAlmostEqual(v.time_of(30), 1.0)
        v.close()

    def test_seek_matches_sequential_read(self):
        import numpy as np
        v = VideoSource(self.vid_path)
        a = v.read_bgr(25)
        v.read_bgr(3)
        b = v.read_bgr(25)
        self.assertTrue(np.array_equal(a, b))  # repeated seeks give the same frame
        v.close()

    def test_export_frame(self):
        v = VideoSource(self.vid_path)
        p = v.export_frame(10, self.dir / "ev")
        self.assertTrue(p.is_file() and p.stat().st_size > 0)
        v.close()


class TestEvents(unittest.TestCase):
    def test_add_ids_and_time(self):
        log = EventLog()
        a = log.add(30, 30.0)
        b = log.add(90, 30.0, visibility_label="Uncertain")
        self.assertEqual((a.event_id, b.event_id), (1, 2))
        self.assertEqual(b.time_s, 3.0)
        self.assertEqual(a.source, "manual")
        self.assertIn("NOT the driver", a.perspective)

    def test_invalid_inputs(self):
        log = EventLog()
        with self.assertRaises(ValueError):
            log.add(-1, 30.0)
        with self.assertRaises(ValueError):
            log.add(1, 0.0)
        with self.assertRaises(ValueError):
            log.add(1, 30.0, visibility_label="Definitely hidden")

    def test_delete_does_not_reuse_ids_of_remaining(self):
        log = EventLog()
        log.add(1, 30.0)
        log.add(2, 30.0)
        self.assertTrue(log.delete(1))
        self.assertFalse(log.delete(99))
        self.assertEqual(log.add(3, 30.0).event_id, 3)

    def test_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            log = EventLog()
            log.add(45, 30.0, notes='comma, "quote" and\nnewline', visibility_label="Partial")
            csv_p = log.to_csv(Path(d) / "e.csv")
            json_p = log.to_json(Path(d) / "e.json", "t.avi")
            with csv_p.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["notes"], 'comma, "quote" and\nnewline')
            back = EventLog.from_json(json_p)
            self.assertEqual(back.events[0].frame, 45)
            self.assertEqual(json.loads(json_p.read_text())["video"], "t.avi")

    def test_empty_log_exports(self):
        with tempfile.TemporaryDirectory() as d:
            log = EventLog()
            log.to_csv(Path(d) / "e.csv")
            self.assertEqual(EventLog.from_json(log.to_json(Path(d) / "e.json")).events, [])


if __name__ == "__main__":
    unittest.main()
