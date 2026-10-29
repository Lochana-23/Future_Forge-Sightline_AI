import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_detect  # noqa: E402


class TestProbe(unittest.TestCase):
    def test_pick_frames_spread_and_in_range(self):
        f = probe_detect.pick_frames(9000, 8)
        self.assertEqual(len(f), 8)
        self.assertTrue(all(0 <= x < 9000 for x in f))
        self.assertEqual(f, sorted(f))

    def test_pick_frames_edge_cases(self):
        self.assertEqual(probe_detect.pick_frames(0, 8), [])
        self.assertEqual(probe_detect.pick_frames(5, 0), [])
        self.assertLessEqual(len(probe_detect.pick_frames(3, 8)), 3)


if __name__ == "__main__":
    unittest.main()
