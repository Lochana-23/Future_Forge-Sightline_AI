import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.candidates import (box_fraction_in_polygon, diagnose, dist_point_to_polygon,  # noqa: E402
                                  explain_frame, find_candidates, iou, load_zones, point_in_polygon,
                                  rider_suspects)
from sightline.tracks import TrackRow  # noqa: E402

ZONE = {"cw": [(100, 100), (300, 100), (300, 200), (100, 200)]}
FPS = 30.0


def car(frame, tid, x, y=150, cls="car", w=60, h=40):
    return TrackRow(frame, frame / FPS, tid, cls, 0.9, x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def ped(frame, tid, x, y=190):     # foot point at (x, y)
    return TrackRow(frame, frame / FPS, tid, "person", 0.8, x - 5, y - 30, x + 5, y)


class TestGeometry(unittest.TestCase):
    def test_point_in_polygon(self):
        sq = ZONE["cw"]
        self.assertTrue(point_in_polygon((150, 150), sq))
        self.assertFalse(point_in_polygon((50, 150), sq))
        self.assertFalse(point_in_polygon((150, 250), sq))
        tri = [(0, 0), (10, 0), (0, 10)]
        self.assertTrue(point_in_polygon((2, 2), tri))
        self.assertFalse(point_in_polygon((8, 8), tri))

    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_zone_loading_validation(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "z.json"
            good.write_text(json.dumps({"zones": [{"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}]}))
            self.assertIn("a", load_zones(good))
            bad = Path(d) / "b.json"
            bad.write_text(json.dumps({"zones": [{"name": "a", "polygon": [[0, 0], [1, 0]]}]}))
            with self.assertRaises(ValueError):
                load_zones(bad)
            empty = Path(d) / "e.json"
            empty.write_text(json.dumps({"zones": []}))
            with self.assertRaises(ValueError):
                load_zones(empty)


class TestCandidates(unittest.TestCase):
    def test_basic_event(self):
        rows = []
        for f in range(0, 30, 3):
            rows += [car(f, 1, 120 + f * 3), ped(f, 9, 200)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].vehicle_tracks, [1])
        self.assertEqual(c[0].pedestrian_tracks, [9])
        self.assertEqual(c[0].zone, "cw")
        self.assertNotIn("vehicle_near_stationary", c[0].flags)

    def test_no_pedestrian_no_event(self):
        rows = [car(f, 1, 150) for f in range(0, 30, 3)]
        self.assertEqual(find_candidates(rows, ZONE, FPS), [])

    def test_pedestrian_outside_zone_no_event(self):
        rows = []
        for f in range(0, 30, 3):
            rows += [car(f, 1, 150), ped(f, 9, 500)]
        self.assertEqual(find_candidates(rows, ZONE, FPS), [])

    def test_single_observation_dropped(self):
        rows = [car(0, 1, 150), ped(0, 9, 200)]
        self.assertEqual(find_candidates(rows, ZONE, FPS, min_obs=2), [])

    def test_stationary_vehicle_flagged(self):
        rows = []
        for f in range(0, 90, 3):
            rows += [car(f, 1, 150), ped(f, 9, 200)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertIn("vehicle_near_stationary", c[0].flags)

    def test_gap_splits_events(self):
        rows = []
        for f in list(range(0, 15, 3)) + list(range(300, 315, 3)):
            rows += [car(f, 1, 150 + (f % 7)), ped(f, 9, 200)]
        self.assertEqual(len(find_candidates(rows, ZONE, FPS, gap_s=1.0)), 2)

    def test_id_switch_is_merged(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 200)]
        for f in range(18, 33, 3):                 # same car, new track id, same place
            rows += [car(f, 2, 152), ped(f, 9, 200)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].vehicle_tracks, [1, 2])
        self.assertIn("merged_multiple_vehicle_tracks", c[0].flags)

    def test_two_different_cars_not_merged(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 130), car(f, 2, 260), ped(f, 9, 200)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertEqual(len(c), 2)
        self.assertEqual(sorted(x.vehicle_tracks[0] for x in c), [1, 2])

    def test_rider_excluded(self):
        rows = []
        for f in range(0, 15, 3):
            m = car(f, 1, 150, cls="motorcycle", w=30, h=40)   # box (135,130)-(165,170)
            rider = TrackRow(f, f / FPS, 9, "person", 0.8, 140, 125, 160, 165)
            rows += [m, rider]
        self.assertEqual(find_candidates(rows, ZONE, FPS), [])
        self.assertTrue(rider_suspects(rows))
        # with the filter off, the rider counts as a pedestrian next to a "vehicle"
        self.assertEqual(len(find_candidates(rows, ZONE, FPS, exclude_riders=False)), 1)

    def test_class_flip_flag_and_evidence_frame(self):
        rows = []
        for f in range(0, 30, 3):
            cls = "bus" if f == 12 else "car"
            rows += [car(f, 1, 120 + f * 2, cls=cls), ped(f, 9, 160 + f)]
        c = find_candidates(rows, ZONE, FPS)[0]
        self.assertIn("vehicle_class_flip", c.flags)
        self.assertIn(c.evidence_frame, range(0, 30, 3))

    def test_far_pedestrian_beyond_gate_gives_no_event(self):
        rows = [r for f in range(0, 15, 3) for r in (car(f, 1, 120), ped(f, 9, 290))]   # ~4.7 ped heights away
        self.assertEqual(find_candidates(rows, ZONE, FPS), [])
        self.assertEqual(len(find_candidates(rows, ZONE, FPS, max_gap_ped_heights=6.0)), 1)

    def test_far_flag_between_two_and_three_heights(self):
        # ped foot (233,190), car box right edge x=180 -> dx=53, dy=20 -> ~57px = 1.9 heights: not flagged
        near = [r for f in range(0, 15, 3) for r in (car(f, 1, 150), ped(f, 9, 233))]
        self.assertNotIn("pedestrian_far_from_vehicle", find_candidates(near, ZONE, FPS)[0].flags)
        # ped foot (250,190): dx=70 -> ~73px = 2.4 heights: flagged but kept
        mid = [r for f in range(0, 15, 3) for r in (car(f, 1, 150), ped(f, 9, 250))]
        c = find_candidates(mid, ZONE, FPS)
        self.assertEqual(len(c), 1)
        self.assertIn("pedestrian_far_from_vehicle", c[0].flags)

    def test_parked_vehicle_gives_one_event_per_pedestrian(self):
        rows = []
        for f in range(0, 30, 3):                       # first pedestrian walks past a parked car
            rows += [car(f, 1, 150), ped(f, 9, 200)]
        for f in range(300, 330, 3):                    # ten seconds later a second one does
            rows += [car(f, 1, 150), ped(f, 12, 200)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertEqual(len(c), 2)
        self.assertEqual([x.pedestrian_tracks for x in c], [[9], [12]])
        self.assertTrue(all("vehicle_near_stationary" in x.flags for x in c))

    def test_pedestrian_id_switch_is_merged(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 200)]
        for f in range(18, 33, 3):                      # same person, new id, same spot
            rows += [car(f, 1, 150), ped(f, 10, 202)]
        c = find_candidates(rows, ZONE, FPS)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].pedestrian_tracks, [9, 10])
        self.assertIn("merged_multiple_pedestrian_tracks", c[0].flags)

    def test_two_different_pedestrians_not_merged(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 130), ped(f, 10, 230)]
        self.assertEqual(len(find_candidates(rows, ZONE, FPS)), 2)

    def test_key_is_stable_and_distinct(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 130), ped(f, 10, 230)]
        a = find_candidates(rows, ZONE, FPS)
        b = find_candidates(rows, ZONE, FPS)
        self.assertEqual([x.key() for x in a], [x.key() for x in b])
        self.assertEqual(len({x.key() for x in a}), 2)

    def test_diagnose_counts(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 200)]     # both in zone, 5 frames
        for f in range(15, 30, 3):
            rows += [car(f, 1, 150)]                      # vehicle only, 5 frames
        d = diagnose(rows, ZONE)
        self.assertEqual(d["cw"]["frames_with_both"], 5)
        self.assertEqual(d["cw"]["frames_with_vehicle"], 10)
        self.assertEqual(d["_processed_frames"], 10)

    def test_explain_frame(self):
        rows = [car(30, 1, 150), ped(30, 9, 200), ped(30, 10, 600)]
        text = "\n".join(explain_frame(rows, ZONE, 30))
        self.assertIn("person#9", text)
        self.assertIn("cw(inside)", text)
        self.assertIn("none", text)                       # person#10 is outside every zone
        self.assertIn("No processed frames", explain_frame(rows, ZONE, 900)[0])

    def test_vehicle_straddling_zone_edge_counts(self):
        # centre y=95 is OUTSIDE the zone (top edge y=100) but ~half the box overlaps it
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150, y=95, h=50), ped(f, 9, 200)]
        self.assertEqual(len(find_candidates(rows, ZONE, FPS)), 1)
        self.assertEqual(len(find_candidates(rows, ZONE, FPS, veh_overlap=0.9)), 0)

    def test_approaching_pedestrian_counts_with_margin_and_is_flagged(self):
        rows = []
        for f in range(0, 15, 3):
            rows += [car(f, 1, 150), ped(f, 9, 60, y=150)]    # 40px left of the zone edge
        c = find_candidates(rows, ZONE, FPS, ped_margin_px=120)
        self.assertEqual(len(c), 1)
        self.assertIn("pedestrian_near_zone_not_inside", c[0].flags)
        self.assertEqual(find_candidates(rows, ZONE, FPS, ped_margin_px=0), [])
        far = [r for f in range(0, 15, 3) for r in (car(f, 1, 150), ped(f, 9, -100, y=150))]
        self.assertEqual(find_candidates(far, ZONE, FPS, ped_margin_px=120), [])

    def test_polygon_helpers(self):
        sq = ZONE["cw"]
        self.assertEqual(dist_point_to_polygon((150, 150), sq), 0.0)
        self.assertAlmostEqual(dist_point_to_polygon((50, 150), sq), 50.0)
        self.assertAlmostEqual(dist_point_to_polygon((0, 0), sq), (100 ** 2 * 2) ** 0.5)
        self.assertEqual(box_fraction_in_polygon((120, 120, 160, 160), sq), 1.0)
        self.assertEqual(box_fraction_in_polygon((400, 400, 440, 440), sq), 0.0)
        self.assertAlmostEqual(box_fraction_in_polygon((80, 120, 120, 160), sq), 0.5, places=1)

    def test_bad_fps(self):
        with self.assertRaises(ValueError):
            find_candidates([], ZONE, 0)


if __name__ == "__main__":
    unittest.main()
