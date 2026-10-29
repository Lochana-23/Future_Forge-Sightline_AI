"""End-to-end regression test of the command-line pipeline on synthetic data (no detector needed)."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sightline.review import ReviewStore  # noqa: E402
from sightline.tracks import TrackRow, write_tracks_csv  # noqa: E402


def run(args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        d = cls.d
        vw = cv2.VideoWriter(str(d / "s.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (400, 300))
        for _ in range(450):
            vw.write(np.full((300, 400, 3), 90, np.uint8))
        vw.release()
        rows = []
        for f in range(0, 60, 3):                                    # car parked, pedestrian 9 passes
            rows += [TrackRow(f, f / 30, 1, "car", 0.9, 120, 130, 180, 170),
                     TrackRow(f, f / 30, 9, "person", 0.8, 195, 160, 205, 190)]
        for f in range(300, 345, 3):                                 # later, pedestrian 12 passes the same car
            rows += [TrackRow(f, f / 30, 1, "car", 0.9, 120, 130, 180, 170),
                     TrackRow(f, f / 30, 12, "person", 0.8, 195, 160, 205, 190)]
        for f in range(150, 200, 3):                                 # a rider on a scooter: must be excluded
            rows += [TrackRow(f, f / 30, 30, "motorcycle", 0.8, 200, 130, 230, 170),
                     TrackRow(f, f / 30, 31, "person", 0.8, 203, 125, 227, 165)]
        write_tracks_csv(rows, d / "t.csv")
        (d / "z.json").write_text(json.dumps({"zones": [{"name": "cw", "polygon": [[100, 100], [350, 100], [350, 220], [100, 220]]}]}))
        (d / "gt.csv").write_text("time_s,note\n1.0,first\n10.5,second\n20.0,missed\n")
        cls.cand = run(["tools/make_candidates.py", str(d / "s.avi"), "--tracks", str(d / "t.csv"),
                        "--zones", str(d / "z.json"), "--out", str(d / "o")])

    def test_candidates_built(self):
        self.assertEqual(self.cand.returncode, 0, self.cand.stderr)
        doc = json.loads((self.d / "o" / "candidates.json").read_text())
        ev = doc["events"]
        self.assertEqual(len(ev), 2, [(e["vehicle_class"], e["pedestrian_tracks"]) for e in ev])
        self.assertEqual({e["pedestrian_tracks"] for e in ev}, {"9", "12"})     # rider 31 excluded
        self.assertTrue(all(e["source"] == "auto" and e["review_status"] == "unreviewed" for e in ev))
        self.assertTrue(all(e["candidate_key"] and Path(e["evidence_sheet"]).is_file() for e in ev))

    def test_review_evaluate_roundtrip(self):
        doc = json.loads((self.d / "o" / "candidates.json").read_text())
        store = ReviewStore(self.d / "rv.json")
        store.set(doc["events"][0]["candidate_key"], "confirmed", "Partial", "stopped/parked vehicle", "ok")
        store.set(doc["events"][1]["candidate_key"], "rejected")
        store.save()
        r = run(["tools/evaluate.py", str(self.d / "s.avi"), "--candidates", str(self.d / "o" / "candidates.json"),
                 "--reviews", str(self.d / "rv.json"), "--gt-csv", str(self.d / "gt.csv"), "--out", str(self.d / "ev")])
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.loads((self.d / "ev" / "evaluation_report.json").read_text())
        self.assertEqual(rep["matching"]["reference_moments"], 3)
        self.assertEqual(rep["matching"]["reference_matched"], 2)               # 20 s is outside every candidate, missed on purpose
        self.assertEqual(rep["human_review"]["confirmed"], 1)
        self.assertEqual(rep["human_review"]["confirmed_share_of_decided"], 0.5)

    def test_explain_frame_mode(self):
        r = run(["tools/make_candidates.py", str(self.d / "s.avi"), "--tracks", str(self.d / "t.csv"),
                 "--zones", str(self.d / "z.json"), "--explain-frame", "30"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("person#9", r.stdout)
        self.assertIn("PAIR", r.stdout)

    def test_bad_inputs_fail_cleanly(self):
        r = run(["tools/make_candidates.py", str(self.d / "s.avi"), "--tracks", str(self.d / "missing.csv"),
                 "--zones", str(self.d / "z.json")])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Could not read", r.stderr)
        r = run(["tools/evaluate.py", str(self.d / "s.avi"), "--candidates", str(self.d / "nope.json")])
        self.assertNotEqual(r.returncode, 0)

    def test_visibility_tool_refuses_example(self):
        r = run(["tools/visibility_check.py", "--calibration", "config/calibration.example.json",
                 "--scenario", "config/scenario.example.json", "--out", str(self.d / "vis")])
        self.assertEqual(r.returncode, 2)
        self.assertIn("NOT_ASSESSABLE", r.stdout)
        self.assertNotIn("Geometric label", r.stdout)


if __name__ == "__main__":
    unittest.main()
