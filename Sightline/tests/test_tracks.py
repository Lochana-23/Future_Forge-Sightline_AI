import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.tracks import TrackRow, read_tracks_csv, segment_frames, summarize, write_tracks_csv  # noqa: E402


def row(frame, tid, cls="car", conf=0.8):
    return TrackRow(frame, frame / 30, tid, cls, conf, 1.0, 2.0, 30.0, 40.0)


class TestTracks(unittest.TestCase):
    def test_segment_frames(self):
        f = segment_frames(60, 61, 30.0, 9000, 3)
        self.assertEqual(f, list(range(1800, 1830, 3)))

    def test_segment_clipped_to_video(self):
        f = segment_frames(299, 400, 30.0, 9000, 5)
        self.assertTrue(max(f) < 9000)

    def test_segment_beyond_end_is_empty(self):
        self.assertEqual(segment_frames(400, 500, 30.0, 9000, 3), [])

    def test_segment_bad_args(self):
        with self.assertRaises(ValueError):
            segment_frames(10, 5, 30.0, 9000, 3)
        with self.assertRaises(ValueError):
            segment_frames(0, 5, 30.0, 9000, 0)
        with self.assertRaises(ValueError):
            segment_frames(0, 5, 0.0, 9000, 3)

    def test_csv_roundtrip_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_tracks_csv([row(1, 7), row(4, 7, "truck")], Path(d) / "t.csv")
            back = read_tracks_csv(p)
            self.assertEqual((back[1].track_id, back[1].cls, back[0].source), (7, "truck", "auto"))
            e = write_tracks_csv([], Path(d) / "e.csv")
            self.assertEqual(read_tracks_csv(e), [])

    def test_summary_flags_instability(self):
        rows = [row(i, 1) for i in range(10)]                       # stable car
        rows += [row(i, 2, "car") for i in range(3)] + [row(5, 2, "truck")]  # short + class flip
        rows += [row(1, -1, "person")]                              # untracked
        s = summarize(rows)
        self.assertEqual(s["unique_tracks"], 2)
        self.assertEqual(s["untracked_detections"], 1)
        self.assertEqual(s["short_tracks_lt_5_obs"], 1)
        self.assertEqual(s["tracks_with_class_flips"], 1)
        self.assertEqual(s["tracks_per_class"], {"car": 2})


if __name__ == "__main__":
    unittest.main()
