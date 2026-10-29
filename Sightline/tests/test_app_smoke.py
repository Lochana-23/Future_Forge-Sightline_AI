"""Smoke test: run app.py top to bottom against a FAKE streamlit.

This catches NameErrors, wrong function names/arguments and broken logic paths. It does NOT prove the real
Streamlit widgets behave identically (that still needs a manual run of `streamlit run app.py`).
"""
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sightline import geometry as geo  # noqa: E402
from sightline.events import EventLog  # noqa: E402


class StopApp(Exception):
    pass


class RerunApp(Exception):
    pass


class FakeSS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):          # columns / sidebar / placeholders delegate to the module-level widgets
        return getattr(self._st, name)


def make_fake(press=(), submit=(), choices=None):
    choices = choices or {}
    st = types.ModuleType("streamlit")
    st.session_state = FakeSS()
    st.calls = []
    ctx = Ctx(st)

    def rec(name):
        def f(*a, **k):
            st.calls.append((name, a, k))
            return ctx
        return f

    for n in ("set_page_config", "title", "caption", "subheader", "write", "image", "warning", "error", "success",
              "info", "json", "markdown", "dataframe", "divider"):
        setattr(st, n, rec(n))

    def keyed(default_fn):
        def f(label, *a, key=None, **k):
            v = st.session_state[key] if key is not None and key in st.session_state else default_fn(label, *a, **k)
            if key is not None:
                st.session_state[key] = v
            return v
        return f

    st.text_input = keyed(lambda label, value="", **k: value)
    st.text_area = keyed(lambda label, value="", **k: value)
    st.number_input = keyed(lambda label, min_value=None, max_value=None, value=None, **k:
                            value if value is not None else (min_value if min_value is not None else 0))
    st.slider = keyed(lambda label, min_value=0, max_value=100, value=None, **k: min_value if value is None else value)
    st.checkbox = keyed(lambda label, value=False, **k: value)
    st.selectbox = keyed(lambda label, options, index=0, **k: choices.get(label, list(options)[index]))
    st.radio = keyed(lambda label, options, index=0, **k: choices.get(label, list(options)[index]))

    def button(label, on_click=None, args=(), kwargs=None, **k):
        if label in press:
            if on_click:
                on_click(*args, **(kwargs or {}))
            return True
        return False

    def form_submit_button(label, on_click=None, args=(), kwargs=None, **k):
        if label in submit:
            if on_click:
                on_click(*args, **(kwargs or {}))
            return True
        return False

    st.button = button
    st.form_submit_button = form_submit_button
    st.columns = lambda spec, **k: [ctx for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [ctx for _ in labels]
    st.form = lambda *a, **k: ctx
    st.expander = lambda *a, **k: ctx
    st.spinner = lambda *a, **k: ctx
    st.empty = lambda: ctx
    st.sidebar = ctx
    st.cache_resource = lambda f: f

    def stop():
        raise StopApp()

    def rerun():
        raise RerunApp()

    st.stop, st.rerun = stop, rerun
    return st


def make_workdir(with_candidates=True, calibration=None):
    d = Path(tempfile.mkdtemp())
    (d / "data").mkdir()
    vw = cv2.VideoWriter(str(d / "data" / "video_006.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for _ in range(150):
        vw.write(np.full((240, 320, 3), 100, np.uint8))
    vw.release()
    if with_candidates:
        log = EventLog()
        for i in range(3):
            log.add(30 + i * 40, 30.0, source="auto", review_status="unreviewed", candidate_key=f"key{i}",
                    frame_start=20 + i * 40, frame_end=50 + i * 40, zone="cw", vehicle_class="car",
                    vehicle_tracks="1", pedestrian_tracks="9", flags="short_event;3_likely_rider_boxes_excluded_in_zone",
                    pedestrian_box="100,100,120,160", model_confidence="0.80 (mean detector score; not correctness)")
        log.to_json(d / "outputs" / "candidates" / "candidates.json", "video_006.avi")
    if calibration is not None:
        (d / "config").mkdir()
        (d / "config" / "calibration.json").write_text(json.dumps(calibration))
    return d


def run_app(workdir, press=(), submit=(), choices=None, st=None):
    st = st or make_fake(press, submit, choices)
    old_cwd, old_mod = os.getcwd(), sys.modules.get("streamlit")
    old_root = os.environ.get("SIGHTLINE_ROOT")
    sys.modules["streamlit"] = st
    os.chdir(workdir)
    os.environ["SIGHTLINE_ROOT"] = str(workdir)
    try:
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        except (StopApp, RerunApp):
            pass
    finally:
        os.chdir(old_cwd)
        if old_root is None:
            os.environ.pop("SIGHTLINE_ROOT", None)
        else:
            os.environ["SIGHTLINE_ROOT"] = old_root
        if old_mod is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old_mod
    return st


def names(st):
    return [c[0] for c in st.calls]


class TestAppSmoke(unittest.TestCase):
    def test_default_render_all_tabs(self):
        st = run_app(make_workdir())
        errors = [str(c[1]) for c in st.calls if c[0] == "error"]
        self.assertEqual(len(errors), 1, errors)                 # only the geometry tab's "not assessable"
        self.assertIn("Not assessable", errors[0])
        self.assertIn("image", names(st))

    def test_missing_video_shows_error_and_stops(self):
        d = make_workdir()
        (d / "data" / "video_006.avi").unlink()
        st = run_app(d)
        self.assertIn("error", names(st))

    def test_navigation_buttons_update_frame(self):
        st = run_app(make_workdir(), press={"+1 s"})
        self.assertEqual(st.session_state.frame, 30)
        self.assertEqual(st.session_state.frame_input, 30)
        self.assertEqual(st.session_state.slider, 30)
        self.assertAlmostEqual(st.session_state.time_input, 1.0)

    def test_navigation_clamps_at_start(self):
        st = run_app(make_workdir(), press={"-10 s"})
        self.assertEqual(st.session_state.frame, 0)

    def test_manual_event_is_saved_to_disk_immediately(self):
        d = make_workdir()
        run_app(d, submit={"Add event"})
        doc = json.loads((d / "outputs" / "events.json").read_text())
        self.assertEqual(len(doc["events"]), 1)
        ev = doc["events"][0]
        self.assertEqual((ev["event_id"], ev["frame"], ev["video"]), (1, 0, "video_006.avi"))
        self.assertTrue((d / ev["evidence_frame"]).is_file())
        self.assertTrue((d / "outputs" / "events.csv").is_file())

    def test_no_candidates_message(self):
        st = run_app(make_workdir(with_candidates=False))
        self.assertTrue(any(c[0] == "info" and "No candidates loaded" in str(c[1]) for c in st.calls))

    def test_review_save_creates_store_and_next_moves(self):
        d = make_workdir()
        run_app(d, press={"Next candidate"}, submit={"Save review"},
                choices={"Decision": "confirmed", "Visibility label (your judgement from this camera view)": "Partial",
                         "Possible obstruction": "stopped/parked vehicle"})
        data = json.loads((d / "outputs" / "reviews.json").read_text())
        self.assertEqual(len(data["reviews"]), 1)
        r = data["reviews"][0]
        self.assertEqual((r["candidate_key"], r["status"], r["visibility_label"], r["obstruction"]),
                         ("key1", "confirmed", "Partial", "stopped/parked vehicle"))   # 'Next' moved to the 2nd one
        self.assertIn("disclaimer", data)

    def test_saving_unreviewed_records_nothing_and_says_so(self):
        d = make_workdir()
        st = run_app(d, submit={"Save review"})
        self.assertTrue(any(c[0] == "info" and "Nothing was recorded" in str(c[1]) for c in st.calls))
        self.assertEqual(json.loads((d / "outputs" / "reviews.json").read_text())["reviews"], [])

    def test_rejected_decision_clears_label(self):
        d = make_workdir()
        run_app(d, submit={"Save review"},
                choices={"Decision": "rejected", "Visibility label (your judgement from this camera view)": "Blocked"})
        r = json.loads((d / "outputs" / "reviews.json").read_text())["reviews"][0]
        self.assertEqual((r["status"], r["visibility_label"]), ("rejected", "Unassessed"))

    def test_review_export(self):
        d = make_workdir()
        run_app(d, press={"Export reviewed events (CSV + JSON)"})
        self.assertTrue((d / "outputs" / "reviewed" / "reviewed_events.csv").is_file())

    def test_corrupt_review_store_is_reported_not_overwritten(self):
        d = make_workdir()
        (d / "outputs").mkdir(exist_ok=True)
        (d / "outputs" / "reviews.json").write_text("{broken")
        st = run_app(d, submit={"Save review"})
        self.assertIn("error", names(st))
        self.assertEqual((d / "outputs" / "reviews.json").read_text(), "{broken")

    def test_geometry_tab_refuses_without_calibration(self):
        st = run_app(make_workdir())
        self.assertTrue(any(c[0] == "error" and "Not assessable" in str(c[1]) for c in st.calls))

    def test_geometry_tab_refuses_bad_calibration(self):
        st = run_app(make_workdir(calibration={"example_only": True}))
        self.assertTrue(any(c[0] == "error" and "does not meet" in str(c[1]) for c in st.calls))

    def test_geometry_tab_runs_with_valid_calibration(self):
        H = np.array([[0.02, 0.003, -5.0], [0.001, 0.025, -8.0], [0.00005, 0.0001, 1.0]])
        img = [[100, 100], [900, 120], [880, 700], [120, 650], [500, 400], [300, 250]]
        cal = {"example_only": False, "source": "synthetic", "units": "metres", "measurement_error_m": 0.2,
               "ground_points": [{"image": p, "world": list(geo.apply_h(H, p))} for p in img]}
        d = make_workdir(calibration=cal)
        st = run_app(d, press={"Run experimental analysis"})
        self.assertTrue((d / "outputs" / "visibility" / "visibility_report.json").is_file())
        rep = json.loads((d / "outputs" / "visibility" / "visibility_report.json").read_text())
        self.assertEqual(rep["status"], "EXPERIMENTAL_OK")
        self.assertIn("EXPERIMENTAL", " ".join(str(c[1]) for c in st.calls if c[0] == "markdown"))


class TestAppPersistenceFlows(unittest.TestCase):
    """Whole-app flows against the fake Streamlit. Each run_app call with a fresh fake is a full app restart."""

    def events(self, d):
        return json.loads((d / "outputs" / "events.json").read_text())["events"]

    def test_two_sessions_and_a_restart_keep_both_events(self):
        d = make_workdir(with_candidates=False)
        run_app(d, press={"+1 s"}, submit={"Add event"})                  # session 1: frame 30
        run_app(d, press={"+1 frame"}, submit={"Add event"})              # new session (restart): frame 1
        ev = self.events(d)
        self.assertEqual([(e["event_id"], e["frame"]) for e in ev], [(1, 30), (2, 1)])
        self.assertNotEqual(ev[0]["evidence_frame"], ev[1]["evidence_frame"])
        self.assertTrue(all((d / e["evidence_frame"]).is_file() for e in ev))
        st = run_app(d)                                                   # a third start only displays them
        self.assertEqual(len(self.events(d)), 2)
        self.assertTrue(any(c[0] == "dataframe" for c in st.calls))

    def test_failed_save_is_reported_and_keeps_what_the_user_typed(self):
        d = make_workdir(with_candidates=False)
        (d / "outputs").write_text("a file where the folder should be")
        st = make_fake(submit={"Add event"})
        st.session_state["ev_notes"] = "my careful notes"
        run_app(d, st=st)
        self.assertEqual(st.session_state.flash[0], "error")
        self.assertIn("NOT SAVED", st.session_state.flash[1])
        self.assertEqual(st.session_state.ev_notes, "my careful notes")

    def test_successful_save_clears_notes_and_reports_the_file(self):
        d = make_workdir(with_candidates=False)
        st = make_fake(submit={"Add event"})
        st.session_state["ev_notes"] = "typed"
        run_app(d, st=st)
        self.assertEqual(st.session_state.flash[0], "success")
        self.assertIn("events.json", st.session_state.flash[1])
        self.assertEqual(st.session_state.ev_notes, "")

    def test_corrupt_events_file_blocks_saving_and_is_not_overwritten(self):
        d = make_workdir(with_candidates=False)
        (d / "outputs").mkdir()
        (d / "outputs" / "events.json").write_text("{oops")
        st = run_app(d, submit={"Add event"})
        self.assertEqual((d / "outputs" / "events.json").read_text(), "{oops")
        self.assertEqual(st.session_state.flash[0], "error")
        self.assertTrue(any(c[0] == "error" and "could not be read" in str(c[1]) for c in st.calls))

    def test_delete_requires_confirmation_then_persists(self):
        d = make_workdir(with_candidates=False)
        run_app(d, submit={"Add event"})
        st = make_fake(press={"Delete event"})
        run_app(d, st=st)
        self.assertEqual(len(self.events(d)), 1)                          # not confirmed: nothing deleted
        st = make_fake(press={"Delete event"})
        st.session_state["del_confirm"] = True
        run_app(d, st=st)
        self.assertEqual(self.events(d), [])
        self.assertTrue(list((d / "outputs" / "backups").glob("events_predelete_*.json")))

    def test_export_button_reports_matching_files(self):
        d = make_workdir(with_candidates=False)
        run_app(d, submit={"Add event"})
        st = run_app(d, press={"Export CSV + JSON (rewrite both from the saved log and check they match)"})
        self.assertEqual(st.session_state.flash[0], "success")
        self.assertEqual(len(self.events(d)), 1)

    def test_switching_video_resets_stale_positions(self):
        d = make_workdir(with_candidates=False)
        st = make_fake()
        st.session_state.update(frame=9999, frame_input=9999, slider=9999, time_input=999.0, loaded_video="other.avi")
        run_app(d, st=st)
        self.assertEqual((st.session_state.frame, st.session_state.slider), (0, 0))

    def test_review_decision_survives_restart_and_export_counts_match(self):
        d = make_workdir()
        run_app(d, submit={"Save review"}, choices={"Decision": "confirmed", "Visibility label (your judgement from "
                                                    "this camera view)": "Partial"})
        st = run_app(d, press={"Export reviewed events (CSV + JSON)"})
        doc = json.loads((d / "outputs" / "reviewed" / "reviewed_events.json").read_text())
        self.assertEqual((doc["summary"]["candidates"], doc["summary"]["confirmed"], doc["summary"]["unreviewed"]),
                         (3, 1, 2))
        self.assertTrue(any(c[0] == "success" and "1 with a human decision, 2 unreviewed" in str(c[1])
                            for c in st.calls))


if __name__ == "__main__":
    unittest.main()
