"""Regression tests for the full-code audit (reviews, regeneration, candidates, tracks, geometry, paths, privacy,
navigation, packaging). All file writes go to temporary folders."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sightline import geometry as g  # noqa: E402
from sightline import paths  # noqa: E402
from sightline.candidates import find_candidates, unique_keys  # noqa: E402
from sightline.events import EventLog  # noqa: E402
from sightline.nav import clamp_frame, step_by_frames, step_by_seconds  # noqa: E402
from sightline.privacy import blur_regions  # noqa: E402
from sightline.review import ReviewStore, export_reviewed, load_candidates, summary  # noqa: E402
from sightline.tracks import TrackRow, duplicate_boxes, summarize, write_tracks_csv  # noqa: E402

ZONE = {"cw": [(100, 100), (300, 100), (300, 200), (100, 200)]}


def tmp():
    return Path(tempfile.mkdtemp())


def cands(spec):
    """spec: list of (key, start, end, veh, ped, zone) -> candidate Events."""
    log = EventLog()
    for key, s, e, v, p, z in spec:
        log.add((s + e) // 2, 30.0, source="auto", review_status="unreviewed", candidate_key=key, frame_start=s, frame_end=e,
                vehicle_tracks=str(v), pedestrian_tracks=str(p), zone=z)
    return log.events


class TestReviewPersistence(unittest.TestCase):
    def setUp(self):
        self.p = tmp() / "outputs" / "reviews.json"
        self.ev = cands([("k1", 10, 60, 1, 9, "cw"), ("k2", 100, 150, 2, 12, "cw"), ("k3", 200, 260, 3, 15, "cw")])

    def test_decision_survives_restart_with_all_fields(self):
        ReviewStore(self.p).decide("k1", "confirmed", "Partial", "stopped/parked vehicle", "Save test", "EW",
                                   event=self.ev[0])
        r = ReviewStore(self.p).get("k1")                                          # a fresh store = restart
        self.assertEqual((r.status, r.visibility_label, r.obstruction, r.notes, r.reviewer),
                         ("confirmed", "Partial", "stopped/parked vehicle", "Save test", "EW"))
        self.assertEqual((r.zone, r.frame_start, r.frame_end, r.vehicle_tracks), ("cw", 10, 60, "1"))

    def test_stale_session_does_not_wipe_other_decisions(self):
        a, b = ReviewStore(self.p), ReviewStore(self.p)
        a.decide("k1", "confirmed", event=self.ev[0])
        b.decide("k2", "rejected", event=self.ev[1])           # b was opened before a saved
        self.assertEqual({k: v.status for k, v in ReviewStore(self.p).reviews.items()},
                         {"k1": "confirmed", "k2": "rejected"})

    def test_previous_version_is_kept_and_old_files_without_descriptions_still_load(self):
        self.p.parent.mkdir(parents=True)
        self.p.write_text(json.dumps({"reviews": [{"candidate_key": "k1", "status": "confirmed",
                                                   "visibility_label": "Partial", "obstruction": "not assessed",
                                                   "notes": "Save test", "reviewer": "", "reviewed_utc": "x"}]}))
        s = ReviewStore(self.p)
        self.assertEqual(s.get("k1").notes, "Save test")
        s.decide("k2", "uncertain", event=self.ev[1])
        prev = json.loads((self.p.parent / "backups" / "reviews_previous.json").read_text())
        self.assertEqual([r["candidate_key"] for r in prev["reviews"]], ["k1"])
        self.assertEqual(ReviewStore(self.p).get("k1").notes, "Save test")

    def test_corrupt_or_wrong_shape_file_raises_and_is_untouched(self):
        self.p.parent.mkdir(parents=True)
        for bad in ("{broken", "[]", '{"reviews": [5]}', '{"reviews": [{"candidate_key": "a", "status": "maybe"}]}'):
            self.p.write_text(bad)
            with self.assertRaises(ValueError):
                ReviewStore(self.p)
            self.assertEqual(self.p.read_text(), bad)

    def test_export_has_reviewed_and_unreviewed_rows_with_separate_human_columns(self):
        s = ReviewStore(self.p)
        s.decide("k1", "confirmed", "Blocked", "other pedestrians", "n", "EW", event=self.ev[0])
        s.decide("k2", "rejected", "Clear", event=self.ev[1])
        out = tmp()
        csv_p, json_p = export_reviewed(self.ev, s, out, "v.avi")
        doc = json.loads(json_p.read_text())
        rows = {r["candidate_key"]: r for r in doc["events"]}
        self.assertEqual(len(rows), 3)
        self.assertEqual((rows["k1"]["human_status"], rows["k1"]["human_visibility_label"]), ("confirmed", "Blocked"))
        self.assertEqual((rows["k2"]["human_status"], rows["k2"]["human_visibility_label"]), ("rejected", "Unassessed"))
        self.assertEqual(rows["k3"]["human_status"], "unreviewed")
        self.assertEqual(rows["k1"]["source"], "auto")                  # automated columns unchanged
        self.assertEqual(rows["k1"]["review_status"], "unreviewed")     # the automated record is not overwritten
        self.assertNotIn("human_zone", rows["k1"])
        self.assertEqual(doc["summary"], {**summary(self.ev, s)})
        self.assertEqual((doc["summary"]["candidates"], doc["summary"]["reviewed"], doc["summary"]["confirmed"],
                          doc["summary"]["rejected"], doc["summary"]["unreviewed"]), (3, 2, 1, 1, 1))
        self.assertIn("human_reviewer", csv_p.read_text(encoding="utf-8-sig").splitlines()[0])


class TestRegeneration(unittest.TestCase):
    def setUp(self):
        self.p = tmp() / "reviews.json"
        self.old = cands([("a", 100, 160, 1, 9, "cw"), ("b", 300, 360, 2, 12, "cw")])
        s = ReviewStore(self.p)
        s.decide("a", "confirmed", "Partial", event=self.old[0])
        s.decide("b", "rejected", event=self.old[1])

    def test_unchanged_candidates_keep_their_reviews(self):
        rep = ReviewStore(self.p).reconcile(self.old)
        self.assertEqual((rep["exact"], rep["proposals"], rep["ambiguous"], rep["unmatched"]), (2, [], [], []))

    def test_changed_start_frame_is_proposed_not_applied(self):
        new = cands([("a2", 90, 170, 1, 9, "cw"), ("b", 300, 360, 2, 12, "cw")])      # a's key changed
        s = ReviewStore(self.p)
        rep = s.reconcile(new)
        self.assertEqual([(x["old_key"], x["new_key"]) for x in rep["proposals"]], [("a", "a2")])
        self.assertIn("a", ReviewStore(self.p).reviews)                              # nothing changed yet
        s.apply_rekeys(rep["proposals"], new)
        r = ReviewStore(self.p)
        self.assertEqual((r.get("a2").status, r.get("a2").visibility_label), ("confirmed", "Partial"))
        self.assertNotIn("a", r.reviews)
        self.assertTrue(list((self.p.parent / "backups").glob("reviews_prerekey_*.json")))

    def test_ambiguous_and_unrelated_matches_are_never_attached(self):
        two = cands([("x1", 90, 170, 1, 9, "cw"), ("x2", 110, 150, 1, 9, "cw")])       # both fit review a
        rep = ReviewStore(self.p).reconcile(two)
        self.assertEqual(rep["proposals"], [])
        self.assertEqual([x["key"] for x in rep["ambiguous"]], ["a"])
        other = cands([("y", 100, 160, 7, 99, "cw"), ("z", 100, 160, 1, 9, "other_zone")])  # other ids / other zone
        rep = ReviewStore(self.p).reconcile(other)
        self.assertEqual(rep["proposals"], [])
        self.assertEqual({u["key"] for u in rep["unmatched"]}, {"a", "b"})

    def test_old_reviews_without_description_are_reported_not_guessed(self):
        self.p.write_text(json.dumps({"reviews": [{"candidate_key": "gone", "status": "confirmed"}]}))
        rep = ReviewStore(self.p).reconcile(self.old)
        self.assertEqual(rep["proposals"], [])
        self.assertIn("cannot re-match", rep["unmatched"][0]["reason"])

    def test_backfill_adds_description_to_old_reviews(self):
        self.p.write_text(json.dumps({"reviews": [{"candidate_key": "a", "status": "confirmed"}]}))
        s = ReviewStore(self.p)
        self.assertEqual(s.backfill(self.old), 1)
        self.assertEqual((s.get("a").zone, s.get("a").frame_start), ("cw", 100))


class TestCandidateRules(unittest.TestCase):
    @staticmethod
    def _rows(ped_id):
        rows = []
        for f in range(0, 60, 3):
            rows.append(TrackRow(f, f / 30, 1, "car", 0.9, 120, 130, 180, 170))
            rows.append(TrackRow(f, f / 30, ped_id, "person", 0.8, 195, 160, 205, 190))
        return rows

    def test_pedestrians_without_track_id_are_skipped_not_merged(self):
        self.assertEqual(len(find_candidates(self._rows(9), ZONE, 30.0)), 1)
        self.assertEqual(find_candidates(self._rows(-1), ZONE, 30.0), [])

    def test_overlapping_candidates_with_a_shared_vehicle_are_flagged(self):
        rows = self._rows(9)
        for f in range(0, 60, 3):
            rows.append(TrackRow(f, f / 30, 10, "person", 0.8, 215, 160, 225, 190))
        cs = find_candidates(rows, ZONE, 30.0)
        self.assertEqual(len(cs), 2)
        self.assertTrue(all("overlaps_candidate_same_vehicle" in c.flags for c in cs))
        self.assertTrue(all("overlaps_candidate_same_pedestrian" not in c.flags for c in cs))

    def test_keys_are_unique_within_a_run(self):
        cs = find_candidates(self._rows(9), ZONE, 30.0)
        self.assertEqual(unique_keys(cs + cs), [cs[0].key(), cs[0].key() + "-2"])

    def test_key_is_deterministic(self):
        self.assertEqual(find_candidates(self._rows(9), ZONE, 30.0)[0].key(),
                         find_candidates(self._rows(9), ZONE, 30.0)[0].key())


class TestTracksSummary(unittest.TestCase):
    def test_duplicate_boxes_detected_and_untracked_zero_is_not_called_accuracy(self):
        rows = [TrackRow(0, 0, 1, "car", .9, 0, 0, 100, 100), TrackRow(0, 0, 2, "car", .9, 2, 2, 100, 100),
                TrackRow(0, 0, 3, "car", .9, 300, 300, 400, 400)]
        self.assertEqual(duplicate_boxes(rows), 1)
        s = summarize(rows)
        self.assertEqual(s["same_frame_duplicate_box_pairs"], 1)
        self.assertEqual(s["untracked_detections"], 0)
        self.assertIn("says nothing about whether the ids or boxes are right", s["note"])
        self.assertIn("identity switches and fragmentation (needs ground truth)", s["not_evaluated"])


class TestGeometryHardening(unittest.TestCase):
    @staticmethod
    def calib(**over):
        H = np.array([[0.02, 0.003, -5.0], [0.001, 0.025, -8.0], [0.00005, 0.0001, 1.0]])
        img = [[100, 100], [900, 120], [880, 700], [120, 650], [500, 400], [300, 250]]
        doc = {"example_only": False, "source": "synthetic test", "units": "metres", "measurement_error_m": 0.2,
               "ground_points": [{"image": p, "world": list(g.apply_h(H, p))} for p in img]}
        doc.update(over)
        return doc

    def test_valid_synthetic_calibration_passes(self):
        self.assertEqual(g.check_calibration(self.calib())["status"], "EXPERIMENTAL_OK")

    def test_nan_and_inf_values_are_refused(self):
        for bad in (float("nan"), float("inf")):
            doc = self.calib()
            doc["ground_points"][2]["world"] = [bad, 1.0]
            r = g.check_calibration(doc)
            self.assertEqual(r["status"], "NOT_ASSESSABLE", bad)
            self.assertNotIn("H", r)
        self.assertEqual(g.check_calibration(self.calib(measurement_error_m=float("nan")))["status"], "NOT_ASSESSABLE")
        self.assertEqual(g.check_calibration(self.calib(measurement_error_m=True))["status"], "NOT_ASSESSABLE")

    def test_points_outside_declared_image_are_refused(self):
        r = g.check_calibration(self.calib(image_size=[500, 500]))
        self.assertEqual(r["status"], "NOT_ASSESSABLE")
        self.assertTrue(any("outside image_size" in x for x in r["reasons"]))

    def test_missing_and_malformed_inputs_are_refused_not_crashed(self):
        for doc in ({}, [], "x", None, {"ground_points": "abc"}, {"example_only": True}):
            self.assertEqual(g.check_calibration(doc)["status"], "NOT_ASSESSABLE")
        rep = g.assess({}, {})
        self.assertEqual(rep["status"], "NOT_ASSESSABLE")
        self.assertNotIn("model_estimate", rep)

    def test_pixels_beyond_the_horizon_are_refused(self):
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0.01, 1.0]])
        with self.assertRaises(ValueError):
            g.apply_h(H, (10, -500))

    def test_scenario_validation(self):
        base = {"eye_ground_px": [100, 100], "eye_height_m": [1.1, 1.5], "pedestrian_foot_px": [200, 200],
                "pedestrian_height_m": [1.4, 1.9], "obstacles": []}
        g.validate_scenario(base)
        for bad in ({"eye_ground_px": [float("nan"), 1]}, {"samples": 0}, {"samples": 10 ** 9}, {"obstacles": [5]},
                    {"obstacles": [{"name": "o", "footprint_px": [[0, 0], [1, 1], ["a", 2]], "height_m": [1, 2]}]},
                    {"eye_height_m": [2, 1]}, {"pedestrian_height_m": [float("inf"), float("inf")]}):
            with self.assertRaises(ValueError, msg=bad):
                g.validate_scenario({**base, **bad})

    def test_setback_distances_are_bounded(self):
        cal = self.calib()
        sc = {"eye_ground_px": [500, 650], "eye_height_m": [1.1, 1.5], "pedestrian_foot_px": [500, 300],
              "pedestrian_height_m": [1.4, 1.9],
              "obstacles": [{"name": "o", "footprint_px": [[480, 500], [540, 500], [540, 540], [480, 540]],
                             "height_m": [1.6, 2.2]}],
              "samples": 40, "setback": {"obstacle": "o", "direction_px": [[500, 500], [600, 500]],
                                         "distances_m": [0, 1e9]}}
        with self.assertRaises(ValueError):
            g.assess(cal, sc)


class TestNavigationBoundaries(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(clamp_frame(-5, 100), 0)
        self.assertEqual(clamp_frame(500, 100), 99)
        self.assertEqual(clamp_frame(5, 1), 0)
        self.assertEqual(step_by_seconds(95, 10, 30.0, 100), 99)
        self.assertEqual(step_by_seconds(5, -10, 30.0, 100), 0)
        self.assertEqual(step_by_seconds(50, 0, 30.0, 100), 50)
        self.assertEqual(step_by_seconds(0, 1, 29.97, 1000), 30)
        self.assertEqual(step_by_frames(0, -1, 100), 0)
        self.assertEqual(step_by_frames(99, 1, 100), 99)


class TestPathsAndPrivacy(unittest.TestCase):
    def test_paths_do_not_depend_on_the_working_directory(self):
        a = paths.resolve("outputs/x.json")
        old = os.getcwd()
        os.chdir(tempfile.mkdtemp())
        try:
            self.assertEqual(paths.resolve("outputs/x.json"), a)
        finally:
            os.chdir(old)
        self.assertEqual(a, ROOT / "outputs" / "x.json")

    def test_root_can_be_redirected_for_tests(self):
        d = tmp()
        os.environ["SIGHTLINE_ROOT"] = str(d)
        try:
            self.assertEqual(paths.OUTPUTS, d.resolve() / "outputs")
            self.assertEqual(paths.to_stored(d / "outputs" / "a.jpg"), "outputs/a.jpg")
        finally:
            del os.environ["SIGHTLINE_ROOT"]

    def test_blur_changes_only_head_and_plate_areas(self):
        img = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        p = TrackRow(0, 0, 1, "person", .9, 50, 20, 90, 180)
        out = blur_regions(img, [p])
        self.assertFalse(np.array_equal(out[20:60, 50:90], img[20:60, 50:90]))        # head blurred
        self.assertTrue(np.array_equal(out[100:180, 50:90], img[100:180, 50:90]))     # body untouched
        self.assertTrue(np.array_equal(out[:, 120:], img[:, 120:]))                   # elsewhere untouched
        self.assertTrue(np.array_equal(img, img.copy()))                              # input not modified in place


class TestPackageHygiene(unittest.TestCase):
    def test_package_script_leaves_out_video_outputs_and_personal_config(self):
        import zipfile
        sys.path.insert(0, str(ROOT / "tools"))
        import make_package
        d = tmp() / "Proj"
        for rel in ("app.py", "data/README.md", "data/video_006.avi", "data/other.mp4", "outputs/.gitkeep",
                    "outputs/events.json", "outputs/evidence/a.jpg", "config/zones.json", "config/calibration.json",
                    "config/calibration.example.json", ".venv/lib/x.py", "sightline/__pycache__/a.pyc",
                    "sightline/store.py", "docs/notes.zip", "tests/t.py"):
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(b"x")
        dest = d.parent / "out.zip"
        make_package.build(d, dest)
        names = set(zipfile.ZipFile(dest).namelist())
        self.assertEqual(names, {"Proj/app.py", "Proj/data/README.md", "Proj/outputs/.gitkeep",
                                 "Proj/config/calibration.example.json", "Proj/sightline/store.py", "Proj/tests/t.py"})

    def test_gitignore_protects_data_and_outputs(self):
        gi = (ROOT / ".gitignore").read_text()
        for line in ("data/*", "!data/README.md", "outputs/*", ".venv/"):
            self.assertIn(line, gi)


class TestRegenerationOfCandidatesTool(unittest.TestCase):
    def test_second_run_keeps_the_first_runs_files_and_reviews_file_is_untouched(self):
        d = tmp()
        vw = cv2.VideoWriter(str(d / "s.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (400, 300))
        for _ in range(90):
            vw.write(np.full((300, 400, 3), 90, np.uint8))
        vw.release()
        rows = []
        for f in range(0, 60, 3):
            rows += [TrackRow(f, f / 30, 1, "car", .9, 120, 130, 180, 170),
                     TrackRow(f, f / 30, 9, "person", .8, 195, 160, 205, 190)]
        write_tracks_csv(rows, d / "t.csv")
        (d / "z.json").write_text(json.dumps({"zones": [{"name": "cw", "polygon": [[100, 100], [350, 100], [350, 220],
                                                                                  [100, 220]]}]}))
        (d / "reviews.json").write_text(json.dumps({"reviews": [{"candidate_key": "x", "status": "confirmed",
                                                                 "notes": "Save test"}]}))
        before = (d / "reviews.json").read_bytes()
        args = [sys.executable, "tools/make_candidates.py", str(d / "s.avi"), "--tracks", str(d / "t.csv"),
                "--zones", str(d / "z.json"), "--out", str(d / "o")]
        r1 = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
        first = (d / "o" / "candidates.json").read_text()
        r2 = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
        self.assertEqual((r1.returncode, r2.returncode), (0, 0), r1.stderr + r2.stderr)
        backups = list((d / "o" / "backups").glob("*/candidates.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text())["events"][0]["candidate_key"],
                         json.loads(first)["events"][0]["candidate_key"])
        self.assertEqual((d / "reviews.json").read_bytes(), before)
        self.assertEqual(load_candidates(d / "o" / "candidates.json")[0].video, "s.avi")
        self.assertIn("NOT anonymised", r2.stdout)


class TestManageEventsTool(unittest.TestCase):
    def test_add_list_check_and_second_add_keeps_the_first(self):
        d = tmp()
        vw = cv2.VideoWriter(str(d / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        for i in range(100):
            vw.write(np.full((48, 64, 3), i, np.uint8))
        vw.release()
        env = {**os.environ, "SIGHTLINE_ROOT": str(d)}

        def run(*a):
            return subprocess.run([sys.executable, "tools/manage_events.py", *a], capture_output=True, text=True,
                                  cwd=ROOT, env=env)
        r1 = run("add", "--video", str(d / "v.avi"), "--frame", "90", "--label", "Partial")
        r2 = run("add", "--video", str(d / "v.avi"), "--frame", "5")
        self.assertEqual((r1.returncode, r2.returncode), (0, 0), r1.stderr + r2.stderr)
        out = run("list").stdout
        self.assertIn("#1  frame 90", out)
        self.assertIn("#2  frame 5", out)
        self.assertEqual(run("check").returncode, 0)
        self.assertNotEqual(run("add", "--video", str(d / "missing.avi"), "--frame", "1").returncode, 0)


if __name__ == "__main__":
    unittest.main()
