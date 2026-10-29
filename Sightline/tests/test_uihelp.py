import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.events import EventLog  # noqa: E402
from sightline.review import ReviewStore  # noqa: E402
from sightline.uihelp import clamp_pos, filter_events, foot_of_box, parse_box, parse_points  # noqa: E402


class TestUiHelp(unittest.TestCase):
    def test_parse_points(self):
        self.assertEqual(parse_points("1,2; 3,4; 5,6"), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(parse_points("1 2\n3 4\n5 6"), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(parse_points("1,2;3,4", min_points=2), [[1, 2], [3, 4]])

    def test_parse_points_errors(self):
        for bad in ("", "1,2; 3,4", "1,2; a,b; 3,4", "1,2,3; 4,5; 6,7"):
            with self.assertRaises(ValueError):
                parse_points(bad)

    def test_boxes(self):
        self.assertEqual(parse_box("1,2,3,4"), (1, 2, 3, 4))
        self.assertIsNone(parse_box(""))
        self.assertIsNone(parse_box("1,2,3"))
        self.assertEqual(foot_of_box("100,100,120,160"), (110.0, 160))
        self.assertIsNone(foot_of_box("x"))

    def test_clamp_pos(self):
        self.assertEqual(clamp_pos(5, 3), 2)
        self.assertEqual(clamp_pos(-1, 3), 0)
        self.assertEqual(clamp_pos("x", 3), 0)
        self.assertEqual(clamp_pos(1, 0), 0)

    def test_filter_events(self):
        import tempfile
        log = EventLog()
        for i, t in enumerate((300, 30, 120)):
            log.add(t, 30.0, candidate_key=f"k{i}")
        st = ReviewStore(Path(tempfile.mkdtemp()) / "r.json")
        st.set("k0", "confirmed", "Clear")
        allv = filter_events(log.events, st, "all")
        self.assertEqual([e.candidate_key for e in allv], ["k1", "k2", "k0"])        # sorted by time
        self.assertEqual([e.candidate_key for e in filter_events(log.events, st, "confirmed")], ["k0"])
        self.assertEqual(len(filter_events(log.events, st, "unreviewed")), 2)
        self.assertEqual(filter_events(log.events, st, "rejected"), [])


if __name__ == "__main__":
    unittest.main()
