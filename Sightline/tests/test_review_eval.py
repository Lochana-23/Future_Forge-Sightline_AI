import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.evaluate import (GtEvent, load_gt_annotations_dir, load_gt_csv, match_events,  # noqa: E402
                                parse_annotation)
from sightline.events import EventLog  # noqa: E402
from sightline.evidence import choose_sheet_frames, make_sheet, parse_ids  # noqa: E402
from sightline.review import (Review, ReviewStore, explain_flags, export_reviewed, summary,  # noqa: E402
                              validate_review)


def make_events(n=3):
    log = EventLog()
    for i in range(n):
        e = log.add(100 + i * 300, 30.0, source="auto", review_status="unreviewed", candidate_key=f"k{i}",
                    frame_start=90 + i * 300, frame_end=130 + i * 300, zone="cw")
    return log.events


class TestReview(unittest.TestCase):
    def test_set_get_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            st = ReviewStore(Path(d) / "r.json")
            st.set("k0", "confirmed", "Partial", "stopped/parked vehicle", "van beside crosswalk", "me")
            st.save()
            back = ReviewStore(Path(d) / "r.json")
            r = back.get("k0")
            self.assertEqual((r.status, r.visibility_label, r.obstruction), ("confirmed", "Partial", "stopped/parked vehicle"))
            self.assertTrue(r.reviewed_utc)
            self.assertEqual(back.get("nope").status, "unreviewed")

    def test_rejected_clears_label(self):
        st = ReviewStore(Path(tempfile.mkdtemp()) / "r.json")
        r = st.set("k1", "rejected", "Blocked", "moving vehicle")
        self.assertEqual((r.visibility_label, r.obstruction), ("Unassessed", "not assessed"))

    def test_invalid_values_rejected(self):
        for bad in (Review("k", "maybe"), Review("k", "confirmed", "Definitely hidden"),
                    Review("k", "confirmed", "Clear", "alien"), Review("")):
            with self.assertRaises(ValueError):
                validate_review(bad)

    def test_unreviewed_removes_entry(self):
        st = ReviewStore(Path(tempfile.mkdtemp()) / "r.json")
        st.set("k0", "confirmed", "Clear")
        st.set("k0", "unreviewed")
        self.assertNotIn("k0", st.reviews)

    def test_corrupt_store_raises_not_silently_empty(self):
        d = Path(tempfile.mkdtemp())
        (d / "r.json").write_text("{not json")
        with self.assertRaises(ValueError):
            ReviewStore(d / "r.json")

    def test_orphans_and_summary(self):
        ev = make_events(3)
        st = ReviewStore(Path(tempfile.mkdtemp()) / "r.json")
        st.set("k0", "confirmed", "Partial")
        st.set("k1", "rejected")
        st.set("gone", "confirmed", "Clear")
        s = summary(ev, st)
        self.assertEqual((s["candidates"], s["confirmed"], s["rejected"], s["unreviewed"]), (3, 1, 1, 1))
        self.assertEqual(s["confirmed_share_of_decided"], 0.5)
        self.assertEqual(s["orphan_reviews"], 1)
        self.assertEqual(s["visibility_labels_among_confirmed"], {"Partial": 1})

    def test_explain_flags(self):
        out = explain_flags("short_event;195_likely_rider_boxes_excluded_in_zone;made_up_flag")
        self.assertEqual(len(out), 3)
        self.assertIn("195", out[1])
        self.assertEqual(out[2], "made_up_flag")
        self.assertEqual(explain_flags(""), [])
        self.assertEqual(explain_flags(None), [])

    def test_export_keeps_auto_and_human_separate(self):
        ev = make_events(2)
        d = Path(tempfile.mkdtemp())
        st = ReviewStore(d / "r.json")
        st.set("k0", "confirmed", "Blocked", "other pedestrians", "x")
        csv_p, json_p = export_reviewed(ev, st, d / "out", "v.avi")
        with csv_p.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(rows[0]["source"], "auto")                  # automated side untouched
        self.assertEqual(rows[0]["review_status"], "unreviewed")     # automated status column not overwritten
        self.assertEqual(rows[0]["human_status"], "confirmed")
        self.assertEqual(rows[1]["human_status"], "unreviewed")
        self.assertIn("disclaimer", json.loads(json_p.read_text()))


class TestEvaluate(unittest.TestCase):
    def test_gt_csv_time_and_frame(self):
        d = Path(tempfile.mkdtemp())
        (d / "g.csv").write_text("time_s,end_time_s,note\n10,12,a\n20,,b\n")
        g = load_gt_csv(d / "g.csv", 30.0)
        self.assertEqual((g[0].start_frame, g[0].end_frame, g[1].start_frame, g[1].end_frame), (300, 360, 600, 600))
        (d / "f.csv").write_text("frame\n450\n")
        self.assertEqual(load_gt_csv(d / "f.csv", 30.0)[0].start_frame, 450)

    def test_gt_csv_errors(self):
        d = Path(tempfile.mkdtemp())
        (d / "e.csv").write_text("time_s\n")
        with self.assertRaises(ValueError):
            load_gt_csv(d / "e.csv", 30.0)
        (d / "b.csv").write_text("note\nhello\n")
        with self.assertRaises(ValueError):
            load_gt_csv(d / "b.csv", 30.0)
        (d / "r.csv").write_text("time_s,end_time_s\n10,5\n")
        with self.assertRaises(ValueError):
            load_gt_csv(d / "r.csv", 30.0)

    def test_annotation_dir_groups_entry_exit(self):
        d = Path(tempfile.mkdtemp())
        (d / "00001300.json").write_text(json.dumps({"class": 1, "points": [1, 2, 3, 4], "group_id": 7}))
        (d / "00001500.json").write_text(json.dumps({"class": 1, "points": [1, 2, 3, 4], "group_id": 7}))
        (d / "00002000.json").write_text(json.dumps({"shapes": [{"label": "0", "group_id": 8, "points": [[0, 0], [1, 1]]}]}))
        (d / "notes.json").write_text("{}")
        (d / "00009999.json").write_text("garbage")
        ev, st = load_gt_annotations_dir(d)
        self.assertEqual([(e.start_frame, e.end_frame, e.cls) for e in ev], [(1300, 1500, 1), (2000, 2000, 0)])
        self.assertEqual((st["parsed"], st["skipped_name"], st["skipped_content"]), (3, 1, 1))

    def test_annotation_dir_empty_gives_helpful_error(self):
        d = Path(tempfile.mkdtemp())
        (d / "00000001.json").write_text(json.dumps({"weird": 1}))
        with self.assertRaises(ValueError) as cm:
            load_gt_annotations_dir(d)
        self.assertIn("weird", str(cm.exception))
        with self.assertRaises(ValueError):
            load_gt_annotations_dir(d / "missing")

    def test_parse_annotation(self):
        self.assertEqual(parse_annotation({"class": "1", "group_id": 3}), (1, 3))
        self.assertIsNone(parse_annotation({"x": 1}))
        self.assertIsNone(parse_annotation([1, 2]))

    def test_match(self):
        ev = make_events(3)           # windows ~ [90,130], [390,430], [690,730] at 30 fps
        gts = [GtEvent("a", 120, 120), GtEvent("b", 1000, 1000, 0), GtEvent("c", 700, 760, 1)]
        m = match_events(ev, gts, 30.0, tol_s=1.0)
        self.assertEqual((m["reference_moments"], m["reference_matched"]), (3, 2))
        self.assertAlmostEqual(m["recall_vs_reference"], 0.667, places=2)
        self.assertEqual(m["unmatched_reference"][0]["id"], "b")
        self.assertEqual(m["candidates_matching_a_reference"], 2)
        self.assertEqual(m["unmatched_candidates"], ["k1"])
        self.assertEqual(m["by_reference_class"], {"0": {"total": 1, "matched": 0}, "1": {"total": 1, "matched": 1}})
        self.assertTrue(m["caveats"])

    def test_match_window_restricts(self):
        ev = make_events(3)
        gts = [GtEvent("a", 120, 120), GtEvent("far", 5000, 5000)]
        m = match_events(ev, gts, 30.0, window_s=(0, 30))
        self.assertEqual(m["reference_moments"], 1)
        self.assertEqual(m["recall_vs_reference"], 1.0)

    def test_match_empty_reference(self):
        m = match_events(make_events(1), [], 30.0)
        self.assertIsNone(m["recall_vs_reference"])


class TestEvidence(unittest.TestCase):
    def test_parse_ids(self):
        self.assertEqual(parse_ids("1;2;7"), [1, 2, 7])
        self.assertEqual(parse_ids(""), [])
        self.assertEqual(parse_ids([3, 4]), [3, 4])

    def test_choose_frames_includes_evidence_and_limits(self):
        frames = range(0, 300, 3)
        sel = choose_sheet_frames(frames, 60, 240, 120, 30.0, n=6)
        self.assertLessEqual(len(sel), 6)
        self.assertIn(120, sel)
        self.assertEqual(choose_sheet_frames([], 0, 10, 5, 30.0), [5])

    def test_make_sheet_shape(self):
        import numpy as np
        imgs = [np.full((100, 200, 3), 50 * i, np.uint8) for i in range(1, 5)]
        sheet = make_sheet(imgs, ["a", "b", "c", "d"], cols=3, thumb_w=100)
        self.assertEqual(sheet.shape[1], 300)
        with self.assertRaises(ValueError):
            make_sheet([], [])


if __name__ == "__main__":
    unittest.main()
