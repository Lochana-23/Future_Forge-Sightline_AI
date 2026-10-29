import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline import geometry as g  # noqa: E402

# A known ground-plane mapping (pixels -> metres) with some perspective, used to make synthetic calibrations.
H_TRUE = np.array([[0.02, 0.003, -5.0], [0.001, 0.025, -8.0], [0.00005, 0.0001, 1.0]])
IMG = [[100, 100], [900, 120], [880, 700], [120, 650], [500, 400], [300, 250], [700, 500]]


def make_calib(noise_m=0.0, n=7, **over):
    rng = np.random.default_rng(0)
    pts = []
    for im in IMG[:n]:
        w = np.array(g.apply_h(H_TRUE, im)) + rng.normal(0, noise_m, 2)
        pts.append({"image": im, "world": [float(w[0]), float(w[1])]})
    doc = {"example_only": False, "source": "measured on satellite imagery", "units": "metres",
           "measurement_error_m": 0.2, "ground_points": pts}
    doc.update(over)
    return doc


IDENT = np.eye(3)   # pixel == metre, for simple scenarios


def scenario(obstacles, **over):
    s = {"eye_ground_px": [0, 0], "eye_height_m": [1.2, 1.2], "eye_jitter_m": 0.0, "pedestrian_foot_px": [10, 0],
         "pedestrian_height_m": [1.7, 1.7], "pixel_pick_sigma_px": 0.0, "obstacles": obstacles, "samples": 40}
    s.update(over)
    return s


WALL = {"name": "wall", "footprint_px": [[5, -1], [5.5, -1], [5.5, 1], [5, 1]], "height_m": [3, 3]}


class TestHomography(unittest.TestCase):
    def test_recovers_known_mapping(self):
        a = np.array(IMG, float)
        w = np.array([g.apply_h(H_TRUE, p) for p in a])
        H = g.fit_homography(a, w)
        for p in ([250, 300], [600, 200], [800, 600]):
            self.assertTrue(np.allclose(g.apply_h(H, p), g.apply_h(H_TRUE, p), atol=1e-6))

    def test_needs_four_points(self):
        with self.assertRaises(ValueError):
            g.fit_homography([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])

    def test_loo_small_for_exact_data(self):
        a = np.array(IMG, float)
        w = np.array([g.apply_h(H_TRUE, p) for p in a])
        self.assertLess(max(g.loo_errors(a, w)), 1e-6)


class TestGate(unittest.TestCase):
    def test_good_calibration_passes(self):
        r = g.check_calibration(make_calib(noise_m=0.05))
        self.assertEqual(r["status"], "EXPERIMENTAL_OK", r["reasons"])
        self.assertIn("H", r)
        self.assertGreaterEqual(r["sigma_cal_m"], 0.2)           # never below the user's stated error

    def test_example_file_refused(self):
        r = g.check_calibration(make_calib(example_only=True))
        self.assertEqual(r["status"], "NOT_ASSESSABLE")
        self.assertNotIn("H", r)

    def test_too_few_points(self):
        self.assertEqual(g.check_calibration(make_calib(n=4))["status"], "NOT_ASSESSABLE")

    def test_missing_fields_refused(self):
        for over in ({"source": ""}, {"units": "feet"}, {"measurement_error_m": 0}, {"measurement_error_m": "x"}):
            self.assertEqual(g.check_calibration(make_calib(**over))["status"], "NOT_ASSESSABLE", over)

    def test_noisy_calibration_refused(self):
        r = g.check_calibration(make_calib(noise_m=3.0))
        self.assertEqual(r["status"], "NOT_ASSESSABLE")
        self.assertTrue(any("too large" in x for x in r["reasons"]))

    def test_collinear_points_refused(self):
        pts = [{"image": [i * 100, i * 50], "world": [i * 2.0, i * 1.0]} for i in range(1, 7)]
        r = g.check_calibration({"example_only": False, "source": "s", "units": "metres",
                                 "measurement_error_m": 0.2, "ground_points": pts})
        self.assertEqual(r["status"], "NOT_ASSESSABLE")

    def test_garbage_input(self):
        self.assertEqual(g.check_calibration([])["status"], "NOT_ASSESSABLE")
        self.assertEqual(g.check_calibration({})["status"], "NOT_ASSESSABLE")
        bad = make_calib()
        bad["ground_points"][0] = {"image": [1]}
        self.assertEqual(g.check_calibration(bad)["status"], "NOT_ASSESSABLE")


class TestLineOfSight(unittest.TestCase):
    def test_segment_blocked_basic(self):
        poly = [(5, -1), (5.5, -1), (5.5, 1), (5, 1)]
        self.assertTrue(g.segment_blocked((0, 0, 1.2), (10, 0, 1.7), poly, 3.0))
        self.assertFalse(g.segment_blocked((0, 0, 1.2), (10, 0, 1.7), poly, 0.5))     # wall lower than the line
        self.assertFalse(g.segment_blocked((0, 5, 1.2), (10, 5, 1.7), poly, 3.0))     # line passes beside the wall

    def test_no_obstacle_is_clear(self):
        r = g.line_of_sight(IDENT, scenario([]), 0.0)
        self.assertEqual(r["geometric_label"], "Clear")
        self.assertEqual(r["visible_body_share_mean"], 1.0)

    def test_tall_wall_is_blocked(self):
        self.assertEqual(g.line_of_sight(IDENT, scenario([WALL]), 0.0)["geometric_label"], "Blocked")

    def test_low_obstacle_gives_partial(self):
        low = dict(WALL, height_m=[1.0, 1.0])
        r = g.line_of_sight(IDENT, scenario([low]), 0.0)
        self.assertEqual(r["geometric_label"], "Partial")
        self.assertTrue(0 < r["visible_body_share_mean"] < 1)

    def test_borderline_is_reported_uncertain_under_sensitivity(self):
        # wall height range straddles the sight line -> samples disagree -> must not claim a confident label
        border = dict(WALL, height_m=[0.2, 3.0])
        r = g.line_of_sight(IDENT, scenario([border], samples=200), 0.0)
        self.assertEqual(r["geometric_label"], "Uncertain")

    def test_eye_inside_obstacle_is_flagged(self):
        box = {"name": "box", "footprint_px": [[-1, -1], [1, -1], [1, 1], [-1, 1]], "height_m": [3, 3]}
        r = g.line_of_sight(IDENT, scenario([box]), 0.0)
        self.assertEqual(r["valid_samples"], 0)
        self.assertEqual(r["geometric_label"], "Uncertain")

    def test_reproducible_with_seed(self):
        s = scenario([dict(WALL, height_m=[0.5, 2.0])], samples=100)
        self.assertEqual(g.line_of_sight(IDENT, s, 0.3), g.line_of_sight(IDENT, s, 0.3))

    def test_scenario_validation(self):
        for bad in ({}, scenario([], eye_height_m=[2, 1]), scenario([{"name": "x", "footprint_px": [[0, 0]],
                                                                       "height_m": 1}])):
            with self.assertRaises(ValueError):
                g.validate_scenario(bad)


class TestWhatIf(unittest.TestCase):
    def test_moving_wall_out_of_line_clears_view(self):
        s = scenario([WALL], setback={"obstacle": "wall", "direction_px": [[0, 0], [0, 1]],
                                      "distances_m": [0, 1, 4]}, samples=30)
        wi = g.what_if_setback(IDENT, s, 0.0)
        labels = [r["geometric_label"] for r in wi["results"]]
        self.assertEqual(labels[0], "Blocked")
        self.assertEqual(labels[-1], "Clear")
        self.assertEqual(wi["label"], "HYPOTHETICAL")
        self.assertTrue(wi["held_constant"] and wi["cannot_predict"])

    def test_bad_setback_inputs(self):
        s = scenario([WALL], setback={"obstacle": "nope", "direction_px": [[0, 0], [0, 1]]})
        with self.assertRaises(ValueError):
            g.what_if_setback(IDENT, s, 0.0)
        s = scenario([WALL], setback={"obstacle": "wall", "direction_px": [[3, 3], [3, 3]]})
        with self.assertRaises(ValueError):
            g.what_if_setback(IDENT, s, 0.0)
        self.assertIsNone(g.what_if_setback(IDENT, scenario([WALL]), 0.0))


class TestAssess(unittest.TestCase):
    def test_refuses_without_valid_calibration(self):
        rep = g.assess(make_calib(example_only=True), scenario([]))
        self.assertEqual(rep["status"], "NOT_ASSESSABLE")
        self.assertNotIn("model_estimate", rep)
        self.assertIn("NOT ASSESSABLE", rep["result"])
        md = g.report_markdown(rep)
        self.assertIn("Not assessable", md)
        self.assertNotIn("Geometric label", md)

    def test_full_report_separates_sections(self):
        s = {"eye_ground_px": [300, 250], "eye_height_m": [1.1, 1.4], "pedestrian_foot_px": [500, 400],
             "pedestrian_height_m": [1.5, 1.9], "obstacles": [], "samples": 20}
        rep = g.assess(make_calib(noise_m=0.05), s)
        self.assertEqual(rep["status"], "EXPERIMENTAL_OK")
        for k in ("measured_by_user", "user_entered_assumptions", "model_estimate"):
            self.assertIn(k, rep)
        self.assertIn("EXPERIMENTAL", g.report_markdown(rep))


if __name__ == "__main__":
    unittest.main()
