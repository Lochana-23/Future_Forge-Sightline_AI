"""Manual-event persistence: the bug where a new session's export replaced earlier events (regression tests).

Everything runs in temporary folders; the real outputs/ folder is never touched. The legacy fixture below mimics
the format the earlier app wrote (no schema_version, no next_id) with an event at frame 2384, labelled Partial.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sightline import store as store_mod  # noqa: E402
from sightline.events import EventLog  # noqa: E402
from sightline.store import EventStore, EventStoreError  # noqa: E402
from sightline.video import VideoSource  # noqa: E402

LEGACY = {"video": "video_006.avi", "note": "old", "events": [{
    "event_id": 1, "frame": 2384, "time_s": 79.467, "source": "manual", "review_status": "manual",
    "event_type": "pedestrian-vehicle interaction", "visibility_label": "Partial",
    "perspective": "Roadside elevated camera view (NOT the driver's viewpoint)", "model_confidence": "",
    "notes": "legacy fixture", "limitations": "x", "evidence_frame": "", "created_utc": "2026-10-20T10:00:00+00:00"}]}


def tmp_outputs():
    return Path(tempfile.mkdtemp()) / "outputs"


def legacy_outputs():
    out = tmp_outputs()
    out.mkdir(parents=True)
    (out / "events.json").write_text(json.dumps(LEGACY), encoding="utf-8")
    return out


def make_video(path, n=60):
    """Frame i is a flat grey whose level encodes i, so an evidence image can be matched to its frame."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), 4 * i, np.uint8))
    vw.release()


def read_json(out):
    return json.loads((out / "events.json").read_text(encoding="utf-8"))


class TestPersistence(unittest.TestCase):
    def test_existing_frame_2384_event_is_loaded(self):
        s = EventStore(legacy_outputs())
        self.assertIsNone(s.blocked)
        self.assertEqual([(e.event_id, e.frame, e.visibility_label, e.video) for e in s.events],
                         [(1, 2384, "Partial", "video_006.avi")])
        self.assertTrue(any("upgraded" in n for n in s.load_notes))

    def test_add_frame_150_keeps_the_old_event_and_gives_distinct_ids(self):
        out = legacy_outputs()
        ev, res = EventStore(out).add(150, 30.0, video="video_006.avi", notes="test")
        self.assertEqual(ev.event_id, 2)
        doc = read_json(out)
        self.assertEqual([(e["event_id"], e["frame"]) for e in doc["events"]], [(1, 2384), (2, 150)])
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(doc["next_id"], 3)

    def test_migration_backs_up_the_original_file_byte_for_byte(self):
        out = legacy_outputs()
        original = (out / "events.json").read_bytes()
        EventStore(out).add(150, 30.0)
        backups = list((out / "backups").glob("events_premigration_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_restart_reloads_both_events(self):
        out = legacy_outputs()
        EventStore(out).add(150, 30.0)
        fresh = EventStore(out)                                    # simulates a full app restart
        self.assertEqual([(e.event_id, e.frame) for e in fresh.events], [(1, 2384), (2, 150)])

    def test_stale_second_session_cannot_overwrite_earlier_events(self):
        out = tmp_outputs()
        a, b = EventStore(out), EventStore(out)                    # two open browser tabs, both loaded empty
        a.add(10, 30.0)
        b.add(20, 30.0)                                            # b must reload before writing
        self.assertEqual([e["frame"] for e in read_json(out)["events"]], [10, 20])
        self.assertEqual([e["event_id"] for e in read_json(out)["events"]], [1, 2])

    def test_many_consecutive_adds(self):
        out = tmp_outputs()
        for i in range(12):
            EventStore(out).add(i * 5, 30.0)
        ids = [e["event_id"] for e in read_json(out)["events"]]
        self.assertEqual(ids, list(range(1, 13)))

    def test_csv_and_json_hold_the_same_events(self):
        out = legacy_outputs()
        s = EventStore(out)
        s.add(150, 30.0, visibility_label="Blocked")
        with (out / "events.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        js = read_json(out)["events"]
        self.assertEqual([(int(r["event_id"]), int(r["frame"]), float(r["time_s"]), r["visibility_label"]) for r in rows],
                         [(e["event_id"], e["frame"], e["time_s"], e["visibility_label"]) for e in js])
        self.assertTrue(s.consistency()["ok"])

    def test_export_refresh_never_drops_events(self):
        out = legacy_outputs()
        s = EventStore(out)
        s.add(150, 30.0)
        fresh = EventStore(out)                                    # a "new session" pressing Export
        fresh.refresh_exports()
        self.assertEqual(len(read_json(out)["events"]), 2)
        self.assertEqual(fresh.consistency()["json_events"], 2)

    def test_consistency_detects_a_stale_csv(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(10, 30.0)
        (out / "events.csv").write_text("event_id,frame,time_s,visibility_label\n", encoding="utf-8")
        self.assertFalse(s.consistency()["ok"])
        s.refresh_exports()
        self.assertTrue(s.consistency()["ok"])

    def test_delete_persists_backs_up_and_never_reuses_the_id(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(10, 30.0)
        s.add(20, 30.0)
        res = s.delete(2)
        self.assertTrue(res.backup.is_file())
        self.assertEqual([e["event_id"] for e in read_json(out)["events"]], [1])
        ev, _ = EventStore(out).add(30, 30.0)
        self.assertEqual(ev.event_id, 3)                           # 2 is NOT reused
        with self.assertRaises(EventStoreError):
            s.delete(99)

    def test_edit_updates_disk_and_rejects_bad_input(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(10, 30.0)
        s.update(1, visibility_label="Blocked", notes="n2")
        self.assertEqual(read_json(out)["events"][0]["visibility_label"], "Blocked")
        with self.assertRaises(EventStoreError):
            s.update(1, visibility_label="Wrong")
        with self.assertRaises(EventStoreError):
            s.update(1, frame=5)                                   # frame is not editable
        with self.assertRaises(EventStoreError):
            s.update(7, notes="x")

    def test_import_skips_duplicates_and_assigns_new_ids(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(150, 30.0)
        src = Path(tempfile.mkdtemp()) / "old.json"
        src.write_text(json.dumps(LEGACY), encoding="utf-8")
        added, skipped, _ = s.import_file(src)
        self.assertEqual((len(added), skipped, added[0].event_id), (1, 0, 2))
        added, skipped, _ = EventStore(out).import_file(src)       # same file again
        self.assertEqual((len(added), skipped), (0, 1))
        self.assertEqual(len(read_json(out)["events"]), 2)


class TestFailureHandling(unittest.TestCase):
    def test_malformed_json_is_never_erased_and_blocks_writes(self):
        out = tmp_outputs()
        out.mkdir(parents=True)
        (out / "events.json").write_text('{"events": [ {"event_id": 1, "fra', encoding="utf-8")
        s = EventStore(out)
        self.assertIsNotNone(s.blocked)
        with self.assertRaises(EventStoreError):
            s.add(10, 30.0)
        self.assertEqual((out / "events.json").read_text(encoding="utf-8"), '{"events": [ {"event_id": 1, "fra')
        self.assertEqual(len(list((out / "backups").glob("events_unreadable_*.json"))), 1)

    def test_incompatible_schemas_are_refused_not_repaired(self):
        for bad in ('[]', '{"events": 5}', '{"schema_version": 99, "events": []}',
                    json.dumps({"events": [{"event_id": 1}]}),                       # missing frame / time
                    json.dumps({"events": [{"event_id": 1, "frame": 1, "time_s": 0.03},
                                           {"event_id": 1, "frame": 2, "time_s": 0.06}]}),   # duplicate ids
                    json.dumps({"events": [{"event_id": 1, "frame": "x", "time_s": 0.1}]})):
            out = tmp_outputs()
            out.mkdir(parents=True)
            (out / "events.json").write_text(bad, encoding="utf-8")
            s = EventStore(out)
            self.assertIsNotNone(s.blocked, bad)
            self.assertEqual((out / "events.json").read_text(encoding="utf-8"), bad)

    def test_explicit_quarantine_moves_the_file_and_starts_fresh(self):
        out = tmp_outputs()
        out.mkdir(parents=True)
        (out / "events.json").write_text("garbage", encoding="utf-8")
        s = EventStore(out)
        dest = s.quarantine_unreadable()
        self.assertEqual(dest.read_text(encoding="utf-8"), "garbage")
        self.assertFalse((out / "events.json").exists())
        self.assertIsNone(s.blocked)
        s.add(5, 30.0)
        self.assertEqual(len(read_json(out)["events"]), 1)

    def test_unwritable_location_reports_failure_and_changes_nothing(self):
        base = Path(tempfile.mkdtemp())
        (base / "outputs").write_text("I am a file, not a folder")        # cannot create events.json inside it
        s = EventStore(base / "outputs")
        with self.assertRaises(EventStoreError) as cm:
            s.add(10, 30.0)
        self.assertIn("Nothing was changed", str(cm.exception))
        self.assertEqual(s.events, [])

    def test_failed_json_write_leaves_memory_and_disk_unchanged(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(10, 30.0)
        before = (out / "events.json").read_bytes()
        with mock.patch.object(store_mod, "atomic_write_text", side_effect=OSError("disk full")):
            with self.assertRaises(EventStoreError):
                s.add(20, 30.0)
        self.assertEqual((out / "events.json").read_bytes(), before)
        self.assertEqual(len(s.events), 1)
        self.assertEqual(EventStore(out).next_id, 2)

    def test_csv_failure_is_reported_but_json_stays_authoritative(self):
        out = tmp_outputs()
        s = EventStore(out)
        with mock.patch.object(store_mod, "atomic_write_csv", side_effect=OSError("locked by Excel")):
            ev, res = s.add(10, 30.0)
        self.assertIn("locked by Excel", res.csv_error)
        self.assertEqual(len(read_json(out)["events"]), 1)
        self.assertFalse(s.consistency()["ok"])                    # and the mismatch is visible
        s.refresh_exports()
        self.assertTrue(s.consistency()["ok"])

    def test_atomic_write_does_not_leave_temp_files_or_half_files(self):
        out = tmp_outputs()
        s = EventStore(out)
        s.add(10, 30.0)
        with mock.patch("sightline.fileio.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(EventStoreError):
                s.add(20, 30.0)
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.suffix == ".tmp"), [])
        self.assertEqual(len(read_json(out)["events"]), 1)

    def test_invalid_input_is_rejected_before_anything_is_written(self):
        out = tmp_outputs()
        s = EventStore(out)
        for kw in ({"visibility_label": "Nope"}, {"source": "auto"}, {"bogus": 1}):
            with self.assertRaises(EventStoreError):
                s.add(10, 30.0, **kw)
        with self.assertRaises(EventStoreError):
            s.add(-1, 30.0)
        with self.assertRaises(EventStoreError):
            s.add(1, 0)
        self.assertFalse((out / "events.json").exists())


class TestEvidencePairing(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        make_video(self.d / "v.avi")
        self.vid = VideoSource(self.d / "v.avi")
        self.out = self.d / "outputs"

    def tearDown(self):
        self.vid.close()

    def add(self, frame, **kw):
        return EventStore(self.out).add(frame, self.vid.fps, video="v.avi",
                                        evidence_writer=lambda dest: self.vid.save_frame(frame, dest), **kw)

    def test_evidence_image_shows_the_event_frame(self):
        ev, _ = self.add(40)
        img = cv2.imread(str(self.d / ev.evidence_frame))
        self.assertAlmostEqual(float(img.mean()), 4 * 40, delta=6)          # frame 40 is grey level 160
        self.assertIn("f000040", ev.evidence_frame)
        self.assertIn(f"event_{ev.event_id:04d}", ev.evidence_frame)

    def test_two_events_get_two_correct_images(self):
        a, _ = self.add(10)
        b, _ = self.add(50)
        self.assertNotEqual(a.evidence_frame, b.evidence_frame)
        for ev, frame in ((a, 10), (b, 50)):
            self.assertAlmostEqual(float(cv2.imread(str(self.d / ev.evidence_frame)).mean()), 4 * frame, delta=6)

    def test_record_failure_removes_the_image_just_written(self):
        with mock.patch.object(store_mod, "atomic_write_text", side_effect=OSError("no space")):
            with self.assertRaises(EventStoreError):
                self.add(10)
        left = list((self.out / "evidence" / "manual").glob("*.jpg")) if (self.out / "evidence").exists() else []
        self.assertEqual(left, [])
        self.assertFalse((self.out / "events.json").exists())

    def test_image_failure_creates_no_record(self):
        def boom(dest):
            Path(dest).write_bytes(b"partial")
            raise OSError("camera exploded")
        with self.assertRaises(EventStoreError) as cm:
            EventStore(self.out).add(10, 30.0, evidence_writer=boom)
        self.assertIn("No event was recorded", str(cm.exception))
        self.assertFalse((self.out / "events.json").exists())
        self.assertEqual(list((self.out / "evidence" / "manual").glob("*")), [])

    def test_existing_image_is_never_overwritten(self):
        ev_dir = self.out / "evidence" / "manual"
        ev_dir.mkdir(parents=True)
        stale = ev_dir / "event_0001_f000010.jpg"
        stale.write_bytes(b"belongs to an old lost event")
        ev, _ = self.add(10)
        self.assertEqual(stale.read_bytes(), b"belongs to an old lost event")
        self.assertNotEqual(ev.evidence_frame, "outputs/evidence/manual/event_0001_f000010.jpg")
        self.assertTrue((self.d / ev.evidence_frame).is_file())

    def test_delete_moves_the_image_instead_of_erasing_it(self):
        ev, _ = self.add(10)
        img = self.d / ev.evidence_frame
        res = EventStore(self.out).delete(ev.event_id)
        self.assertFalse(img.exists())
        self.assertEqual(len(list((self.out / "deleted_evidence").glob("*.jpg"))), 1)
        self.assertTrue(res.notes)

    def test_stored_paths_are_relative_and_work_from_any_cwd(self):
        ev, _ = self.add(10)
        self.assertFalse(os.path.isabs(ev.evidence_frame))
        old = os.getcwd()
        os.chdir(tempfile.mkdtemp())
        try:
            self.assertIsNotNone(EventStore(self.out).evidence_path(ev))
        finally:
            os.chdir(old)


class TestEventLogLoading(unittest.TestCase):
    def test_candidates_loader_rejects_bad_rows_with_a_clear_message(self):
        p = Path(tempfile.mkdtemp()) / "c.json"
        p.write_text(json.dumps({"events": [{"event_id": 1, "frame": 5}]}), encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            EventLog.from_json(p)
        self.assertIn("time_s", str(cm.exception))

    def test_roundtrip_is_lossless(self):
        log = EventLog()
        log.add(5, 30.0, notes="a, \"quoted\"\nline", zone="z")
        p = Path(tempfile.mkdtemp()) / "c.json"
        log.to_json(p, "v.avi")
        self.assertEqual(EventLog.from_json(p).rows(), log.rows())


if __name__ == "__main__":
    unittest.main()
