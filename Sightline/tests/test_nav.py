import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.nav import clamp_frame, step_by_frames, step_by_seconds  # noqa: E402


class TestNav(unittest.TestCase):
    def test_plus_minus_one_second(self):
        self.assertEqual(step_by_seconds(100, 1, 30.0, 9000), 130)
        self.assertEqual(step_by_seconds(100, -1, 30.0, 9000), 70)

    def test_clamped_at_ends(self):
        self.assertEqual(step_by_seconds(10, -1, 30.0, 9000), 0)
        self.assertEqual(step_by_seconds(8990, 10, 30.0, 9000), 8999)
        self.assertEqual(step_by_frames(0, -1, 9000), 0)
        self.assertEqual(step_by_frames(8999, 1, 9000), 8999)

    def test_non_integer_fps(self):
        self.assertEqual(step_by_seconds(0, 1, 29.97, 9000), 30)

    def test_clamp(self):
        self.assertEqual(clamp_frame(-5, 60), 0)
        self.assertEqual(clamp_frame(500, 60), 59)


if __name__ == "__main__":
    unittest.main()
