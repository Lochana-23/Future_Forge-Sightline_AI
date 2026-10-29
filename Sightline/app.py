"""SightLine AI - review workbench.

Run from the project folder:  streamlit run app.py

Tabs:
  1. Video and manual log  - play/seek, export frames, log events by hand
  2. Review candidates     - confirm / reject / mark uncertain, label, note obstruction, export
  3. Experimental geometry - gated line-of-sight check (needs a real calibration; refuses otherwise)

Navigation design (tab 1): the widgets' own session-state values are the single source of truth. Every control
changes them together via sync_widgets(), which may only be called from a callback or before the widgets exist.
"""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from sightline import geometry
from sightline.events import LIMITATIONS_DEFAULT, PERSPECTIVE_DEFAULT, VISIBILITY_LABELS
from sightline.fileio import atomic_write_text
from sightline.nav import step_by_frames, step_by_seconds
from sightline.paths import CONFIG, OUTPUTS, resolve
from sightline.store import EventStore, EventStoreError
from sightline.review import (DISCLAIMER, OBSTRUCTIONS, STATUSES, ReviewStore, explain_flags, export_reviewed,
                              load_candidates, summary)
from sightline.uihelp import clamp_pos, filter_events, foot_of_box, parse_points
from sightline.video import VideoError, VideoSource

OUT = OUTPUTS                      # fixed to the project folder, not to whatever folder Streamlit was started in
EVIDENCE = OUT / "evidence"
REVIEWS = OUT / "reviews.json"
ss = st.session_state

st.set_page_config(page_title="SightLine AI", layout="wide")
st.title("SightLine AI - review workbench")
st.caption("Development sample only. Not evidence of real-world road conditions. "
           "Labels are reviewer-defined; the camera view is not the driver's view.")


@st.cache_resource
def load_video(path: str):
    return VideoSource(path)


def V():
    return load_video(str(resolve(ss.video_path)))


def sync_widgets(frame: int):
    """Set the frame and every widget that shows it. Callback / pre-widget use only."""
    v = V()
    f = v.clamp(frame)
    ss.frame = f
    ss.frame_input = f
    ss.slider = f
    ss.time_input = round(v.time_of(f), 2)


def nudge_seconds(sec):
    v = V()
    sync_widgets(step_by_seconds(ss.frame, sec, v.fps, v.frame_count))


def nudge_frames(n):
    sync_widgets(step_by_frames(ss.frame, n, V().frame_count))


def on_frame_input():
    sync_widgets(ss.frame_input)


def on_slider():
    sync_widgets(ss.slider)


def on_time_input():
    sync_widgets(V().frame_of(ss.time_input))


def flash(kind: str, text: str):
    ss.flash = (kind, text)


def show_flash():
    f = ss.get("flash")
    if f:
        getattr(st, f[0])(f[1])
        ss.flash = None


def add_event_cb():
    """Save a manual event (and its evidence image) immediately. Form inputs are kept if saving fails."""
    v = V()
    frame = int(ss.frame)
    writer = (lambda dest: v.save_frame(frame, dest)) if ss.ev_save_evidence else None
    try:
        ev, res = EventStore(OUT).add(frame, v.fps, video=Path(ss.video_path).name, evidence_writer=writer,
                                      event_type=ss.ev_type, visibility_label=ss.ev_label, notes=ss.ev_notes)
    except EventStoreError as e:
        flash("error", f"NOT SAVED. {e}")
        return
    msg = f"Saved event #{ev.event_id} at frame {frame} ({ev.time_s:.2f} s) to {res.json_path}."
    if ev.evidence_frame:
        msg += f" Evidence image: {ev.evidence_frame}."
    if res.csv_error:
        flash("warning", msg + " WARNING: " + res.csv_error)
    else:
        flash("success", msg)
    ss.ev_notes = ""


def delete_event_cb():
    if not ss.get("del_confirm"):
        flash("warning", "Tick the confirmation box first. Nothing was deleted.")
        return
    try:
        res = EventStore(OUT).delete(int(ss.del_id))
    except EventStoreError as e:
        flash("error", f"NOT DELETED. {e}")
        return
    flash("success", f"Deleted event #{ss.del_id}. A backup of the previous file is in {res.backup}. "
                     + " ".join(res.notes))
    ss.del_confirm = False


def edit_event_cb(eid: int):
    try:
        EventStore(OUT).update(eid, event_type=ss[f"edit_type_{eid}"], visibility_label=ss[f"edit_label_{eid}"],
                               notes=ss[f"edit_notes_{eid}"])
    except EventStoreError as e:
        flash("error", f"NOT SAVED. {e}")
        return
    flash("success", f"Updated event #{eid}. The previous file was backed up.")


def refresh_exports_cb():
    try:
        st_ = EventStore(OUT)
        res = st_.refresh_exports()
        chk = st_.consistency()
    except EventStoreError as e:
        flash("error", f"NOT EXPORTED. {e}")
        return
    if res.csv_error or not chk["ok"]:
        flash("warning", (res.csv_error or "") + f" Consistency problems: {chk['problems']}")
    else:
        flash("success", f"Wrote {res.json_path} and {res.csv_path}: {chk['json_events']} events in each, "
                         "identical ids, frames, times and labels. (They are also updated after every save.)")


def import_cb():
    path = ss.get("import_path", "").strip()
    try:
        added, skipped, res = EventStore(OUT).import_file(resolve(path))
    except EventStoreError as e:
        flash("error", f"NOT IMPORTED. {e}")
        return
    flash("success", f"Imported {len(added)} event(s) with new ids; skipped {skipped} that were already present.")


def quarantine_cb():
    if not ss.get("q_confirm"):
        flash("warning", "Tick the box first. Nothing was moved.")
        return
    try:
        dest = EventStore(OUT).quarantine_unreadable()
    except EventStoreError as e:
        flash("error", str(e))
        return
    flash("success", f"The unreadable file was moved (not deleted) to {dest}. A new empty log will be started.")


# ---- sidebar: load video ---------------------------------------------------
st.sidebar.text_input("Video path (relative to the project folder)", "data/video_006.avi", key="video_path")
try:
    vid = V()
except VideoError as e:
    st.error(str(e))
    st.stop()

st.sidebar.write(f"{vid.width}x{vid.height} | {vid.fps:.2f} FPS | {vid.frame_count} frames | "
                 f"{vid.duration_s:.1f} s")
st.sidebar.caption("Dataset licence: educational, non-commercial, no redistribution. Do not share frames publicly.")

if "frame" not in ss or ss.get("loaded_video") != ss.video_path:   # first run, or the video was changed
    ss.loaded_video = ss.video_path
    sync_widgets(0)
if ss.get("pending") is not None:          # set by Play; applied before widgets are created
    sync_widgets(ss.pending)
    ss.pending = None

tab_video, tab_review, tab_geo = st.tabs(["Video and manual log", "Review candidates", "Experimental: line of sight"])


# =============================== TAB 1: video + manual log ===================================
def video_tab():
    c = st.columns(7)
    c[0].button("-10 s", on_click=nudge_seconds, args=(-10,))
    c[1].button("-1 s", on_click=nudge_seconds, args=(-1,))
    c[2].button("-1 frame", on_click=nudge_frames, args=(-1,))
    c[3].button("+1 frame", on_click=nudge_frames, args=(1,))
    c[4].button("+1 s", on_click=nudge_seconds, args=(1,))
    c[5].button("+10 s", on_click=nudge_seconds, args=(10,))
    play = c[6].button("Play 5 s")

    j1, j2 = st.columns(2)
    j1.number_input("Go to frame", min_value=0, max_value=vid.frame_count - 1, step=1,
                    key="frame_input", on_change=on_frame_input)
    j2.number_input("...or go to time (s)", min_value=0.0, max_value=float(vid.duration_s), step=0.5,
                    key="time_input", on_change=on_time_input)
    st.slider("Frame", 0, vid.frame_count - 1, key="slider", on_change=on_slider)

    placeholder = st.empty()
    if play:
        # Coarse preview: every 5th frame (~6 fps) for 5 seconds. Not real-time playback.
        start = ss.frame
        last = start
        for f in range(start, vid.clamp(start + int(5 * vid.fps)) + 1, 5):
            try:
                placeholder.image(vid.read_rgb(f), caption=f"frame {f} | {vid.time_of(f):.2f} s", width=700)
            except VideoError as e:
                st.warning(str(e))
                break
            last = f
            time.sleep(0.15)
        ss.pending = last
        st.rerun()
    else:
        try:
            placeholder.image(vid.read_rgb(ss.frame),
                              caption=f"frame {ss.frame} | {vid.time_of(ss.frame):.2f} s", width=700)
        except VideoError as e:
            st.error(str(e))

    if st.button("Export this frame as JPEG (plain export, not linked to an event)"):
        try:
            p = vid.export_frame(ss.frame, EVIDENCE)
            st.success(f"Saved {p}")
        except VideoError as e:
            st.error(f"Not saved: {e}")

    st.subheader("Log a manual event at this frame")
    show_flash()
    st.info(f"The event will be recorded at frame {ss.frame} ({vid.time_of(ss.frame):.2f} s), the frame shown above. "
            "It is saved to disk the moment you press Add event.")
    with st.form("event_form", clear_on_submit=False):
        st.text_input("Event type", "pedestrian-vehicle interaction", key="ev_type")
        st.selectbox("Visibility label (reviewer judgement)", VISIBILITY_LABELS, key="ev_label")
        st.caption(f"Perspective: {PERSPECTIVE_DEFAULT}")
        st.text_area("Notes", key="ev_notes")
        st.checkbox("Also save evidence frame (the exact frame shown above)", value=True, key="ev_save_evidence")
        st.form_submit_button("Add event", on_click=add_event_cb)

    st.subheader("Manual event log")
    store = EventStore(OUT)
    if store.blocked:
        st.error(store.blocked)
        st.checkbox("I understand: move the unreadable file aside (it is kept in outputs/backups)", key="q_confirm")
        st.button("Move it aside and start a new log", on_click=quarantine_cb)
        return
    for note in store.load_notes:
        st.info(note)
    if store.events:
        cols = ["event_id", "video", "frame", "time_s", "visibility_label", "event_type", "notes",
                "evidence_frame", "created_utc"]
        st.dataframe(pd.DataFrame([{c: getattr(e, c) for c in cols} for e in store.events]))
        st.caption(f"{len(store.events)} event(s) stored in {store.json_path} (next id: {store.next_id}).")
        ids = [e.event_id for e in store.events]
        sel = st.selectbox("Event", ids, key="sel_event")
        cur = next(e for e in store.events if e.event_id == sel)
        img = store.evidence_path(cur)
        if img:
            st.image(str(img), caption=f"Evidence for event #{cur.event_id}: frame {cur.frame}, {cur.time_s:.2f} s",
                     width=500)
        elif cur.evidence_frame:
            st.warning(f"Evidence image {cur.evidence_frame} is missing on disk.")
        with st.expander("Edit this event"):
            with st.form(f"edit_form_{sel}"):
                st.text_input("Event type", cur.event_type, key=f"edit_type_{sel}")
                st.selectbox("Visibility label", VISIBILITY_LABELS, index=VISIBILITY_LABELS.index(cur.visibility_label),
                             key=f"edit_label_{sel}")
                st.text_area("Notes", cur.notes, key=f"edit_notes_{sel}")
                st.form_submit_button("Save changes", on_click=edit_event_cb, args=(sel,))
        with st.expander("Delete this event"):
            st.checkbox("Yes, remove this event from the log (a backup is kept; its evidence image is moved "
                        "to outputs/deleted_evidence, not erased)", key="del_confirm")
            ss.del_id = sel
            st.button("Delete event", on_click=delete_event_cb)
        d1, _ = st.columns(2)
        d1.button("Export CSV + JSON (rewrite both from the saved log and check they match)",
                  on_click=refresh_exports_cb)
        chk = store.consistency()
        if not chk["ok"]:
            st.warning(f"events.csv and events.json do not match: {chk['problems']}. Press Export to rewrite them.")
    else:
        st.info("No events yet.")
    with st.expander("Import events from another events.json (for example a backup)"):
        st.text_input("Path to an events JSON file", key="import_path")
        st.button("Import (duplicates are skipped; imported events get new ids)", on_click=import_cb)
    st.caption(LIMITATIONS_DEFAULT)
    st.caption("Evidence frames are raw video frames: faces and number plates may be visible. Do not share them; "
               "the dataset licence forbids redistribution.")


# =============================== TAB 2: review candidates ===================================
def apply_rekeys_cb(proposals, events):
    try:
        store = ReviewStore(REVIEWS)
        bk = store.apply_rekeys(proposals, events)
    except (OSError, ValueError) as e:
        flash("error", f"Reviews NOT moved: {e}")
        return
    flash("success", f"Moved {len(proposals)} review(s) to the matching regenerated candidates. Backup: {bk}")


def reconcile_panel(events, store):
    """Candidates may have been regenerated. Saved reviews are re-attached only when the match is unambiguous
    AND the reviewer approves; everything else is reported."""
    try:
        if store.backfill(events):
            store.save()                       # adds identity details to old reviews; decisions are unchanged
    except OSError as e:
        st.warning(f"Could not update review details: {e}")
    rep = store.reconcile(events)
    show_flash()
    if rep["proposals"]:
        st.warning(f"{len(rep['proposals'])} saved review(s) belong to candidates that were regenerated and can be "
                   "re-attached (same zone, overlapping time, shared track ids):")
        st.dataframe(pd.DataFrame(rep["proposals"]))
        st.button("Re-attach these reviews to the new candidates (a backup is made first)",
                  on_click=apply_rekeys_cb, args=(rep["proposals"], events))
    if rep["ambiguous"]:
        st.warning(f"{len(rep['ambiguous'])} saved review(s) could match more than one candidate. They were NOT "
                   "attached to any. Review those candidates again.")
    if rep["unmatched"]:
        st.warning(f"{len(rep['unmatched'])} saved review(s) match no current candidate and are kept in "
                   f"{REVIEWS} but not shown: " + "; ".join(f"{u['key']} ({u['reason']})" for u in rep["unmatched"]))


def review_tab():
    st.caption("Automated candidates are PROPOSALS. Nothing here is a confirmed event until you decide. "
               "Your decisions are saved separately from the automated records.")
    cpath = st.text_input("Candidates file", "outputs/candidates/candidates.json", key="cand_path")
    try:
        events = load_candidates(resolve(cpath))
    except (OSError, ValueError) as e:
        st.info(f"No candidates loaded ({e}). Run tools\\make_candidates.py first, then reload this page.")
        return
    if not events:
        st.info("The candidates file contains no events.")
        return
    try:
        store = ReviewStore(REVIEWS)
    except (OSError, ValueError) as e:
        st.error(f"Could not read {REVIEWS}: {e}. Fix or rename that file; it was NOT overwritten.")
        return

    reconcile_panel(events, store)
    with st.expander("How to use the labels (read once)"):
        st.markdown(
            "**Perspective:** you are judging from the elevated roadside camera, **not** the driver's seat.\n\n"
            "- **Clear** - in this image, nothing hides the pedestrian from the approaching vehicle's side.\n"
            "- **Partial** - part of the pedestrian is hidden by an object in this image.\n"
            "- **Blocked** - the pedestrian is (almost) entirely hidden in this image.\n"
            "- **Uncertain** - you cannot tell from this view. Use this freely.\n\n"
            "These are your judgements. They do not show what any driver saw, noticed or would have done.")

    flt = st.selectbox("Show", ["all"] + STATUSES, key="rv_filter")
    shown = filter_events(events, store, flt)
    if not shown:
        st.info("No candidates match this filter.")
        return
    pos = clamp_pos(ss.get("rv_pos", 0), len(shown))
    n1, n2, n3 = st.columns([1, 1, 6])
    if n1.button("Previous candidate"):
        pos = clamp_pos(pos - 1, len(shown))
    if n2.button("Next candidate"):
        pos = clamp_pos(pos + 1, len(shown))
    ss.rv_pos = pos
    ev = shown[pos]
    cur = store.get(ev.candidate_key)

    st.subheader(f"Candidate {pos + 1} of {len(shown)}  |  {ev.time_s:.1f} s  |  zone: {ev.zone}  |  "
                 f"current decision: {cur.status}")
    st.write(f"Vehicle: **{ev.vehicle_class}** (track {ev.vehicle_tracks}) | Pedestrian track(s): "
             f"{ev.pedestrian_tracks} | Frames {ev.frame_start}-{ev.frame_end} | Detector: {ev.model_confidence}")
    st.caption(ev.notes)
    for line in explain_flags(ev.flags):
        st.warning(line)

    def _img(p):
        return str(resolve(p)) if p and resolve(p).is_file() else None
    img = _img(ev.evidence_sheet) or _img(ev.evidence_frame)
    if img:
        st.image(img, caption="Evidence: yellow = crosswalk zone, red = pedestrian track(s), green = vehicle track(s)",
                 width=1200)
    else:
        try:
            st.image(vid.read_rgb(int(ev.frame)), caption=f"frame {ev.frame}", width=900)
        except VideoError as e:
            st.error(str(e))

    with st.form(f"rv_form_{ev.candidate_key}"):
        status = st.radio("Decision", STATUSES, index=STATUSES.index(cur.status), horizontal=True,
                          help="confirmed = a real vehicle/pedestrian moment; rejected = not a real moment "
                               "(false alarm); uncertain = cannot tell.")
        label = st.selectbox("Visibility label (your judgement from this camera view)", VISIBILITY_LABELS,
                             index=VISIBILITY_LABELS.index(cur.visibility_label))
        obstruction = st.selectbox("Possible obstruction", OBSTRUCTIONS, index=OBSTRUCTIONS.index(cur.obstruction))
        notes = st.text_area("Notes", value=cur.notes)
        reviewer = st.text_input("Reviewer name or initials", value=cur.reviewer)
        if st.form_submit_button("Save review"):
            try:
                r = store.decide(ev.candidate_key, status, label, obstruction, notes, reviewer, event=ev)
                if status == "unreviewed":
                    st.info("Nothing was recorded: the decision is still 'unreviewed'. Choose confirmed, rejected "
                            "or uncertain, then save. (Saving 'unreviewed' clears any earlier decision.)")
                else:
                    st.success(f"Saved: {r.status}" + ("" if status in ("confirmed", "uncertain")
                                                        else " (label and obstruction are cleared for this decision)"))
            except (OSError, ValueError) as e:
                st.error(f"Could not save: {e}")
    st.caption(DISCLAIMER)

    st.divider()
    st.subheader("Review progress")
    s = summary(events, store)
    st.json(s)
    if st.button("Export reviewed events (CSV + JSON)"):
        try:
            csv_p, json_p = export_reviewed(events, store, OUT / "reviewed", Path(ss.video_path).name)
            st.success(f"Saved {csv_p} and {json_p}: {len(events)} candidates, "
                       f"{s['reviewed']} with a human decision, {s['unreviewed']} unreviewed.")
        except OSError as e:
            st.error(f"Not exported: {e}")


# =============================== TAB 3: experimental geometry ===================================
def geometry_tab():
    st.warning("EXPERIMENTAL. " + geometry.DISCLAIMER)
    cal_path = CONFIG / "calibration.json"
    if not cal_path.is_file():
        st.error("Not assessable: config/calibration.json does not exist. This analysis needs at least "
                 f"{geometry.GATE_MIN_POINTS} ground points measured OUTSIDE the video (for example crosswalk "
                 "dimensions from satellite imagery). Copy config/calibration.example.json and fill in real "
                 "values (see docs/GEOMETRY_GUIDE.md). Until then no visibility numbers are produced.")
        return
    try:
        cal = geometry.load_json(cal_path)
    except (OSError, ValueError) as e:
        st.error(f"Could not read {cal_path}: {e}")
        return
    gate = geometry.check_calibration(cal)
    if gate["status"] != "EXPERIMENTAL_OK":
        st.error("Not assessable: the calibration does not meet the minimum requirements.")
        for r in gate["reasons"]:
            st.write(f"- {r}")
        return
    st.success(f"Calibration passed the gate: {gate['n_points']} points, leave-one-out error mean "
               f"{gate['loo_error_m']['mean']} m / max {gate['loo_error_m']['max']} m. This says the points are "
               "self-consistent, not that they are correct.")

    path = ss.get("cand_path", "outputs/candidates/candidates.json")
    try:
        events = load_candidates(resolve(path))
        if events:
            def prefill():
                f = foot_of_box(events[ss.geo_idx].pedestrian_box)
                if f:
                    ss.geo_px, ss.geo_py = float(f[0]), float(f[1])
            st.selectbox("Prefill the pedestrian position from candidate", range(len(events)),
                         format_func=lambda i: f"#{events[i].event_id} at {events[i].time_s:.1f}s", key="geo_idx",
                         on_change=prefill)
    except (OSError, ValueError):
        pass

    st.markdown("**Everything below is YOUR assumption, entered in the same image coordinates as the calibration.**")
    g1, g2 = st.columns(2)
    ex = g1.number_input("Driver eye point on the ground: x (px)", value=600.0, key="geo_ex")
    ey = g1.number_input("Driver eye point on the ground: y (px)", value=700.0, key="geo_ey")
    px = g2.number_input("Pedestrian foot: x (px)", value=600.0, key="geo_px")
    py = g2.number_input("Pedestrian foot: y (px)", value=500.0, key="geo_py")
    h1, h2 = st.columns(2)
    eye_lo = h1.number_input("Eye height low (m)", value=1.1, min_value=0.3, key="geo_eh0")
    eye_hi = h1.number_input("Eye height high (m)", value=1.5, min_value=0.3, key="geo_eh1")
    ped_lo = h2.number_input("Pedestrian height low (m)", value=1.4, min_value=0.3, key="geo_ph0")
    ped_hi = h2.number_input("Pedestrian height high (m)", value=1.9, min_value=0.3, key="geo_ph1")
    o_name = st.text_input("Obstacle name (leave the footprint empty for none)", "obstacle", key="geo_on")
    o_pts = st.text_area("Obstacle ground footprint, image pixels: 'x,y; x,y; x,y; ...'", "", key="geo_op")
    o_lo = st.number_input("Obstacle height low (m)", value=1.6, min_value=0.1, key="geo_oh0")
    o_hi = st.number_input("Obstacle height high (m)", value=2.2, min_value=0.1, key="geo_oh1")
    wi = st.text_input("Optional what-if: move the obstacle along a ground direction. Two points 'x,y; x,y'",
                       "", key="geo_wi")
    dists = st.text_input("What-if distances (metres, comma separated)", "0, 1, 2, 3", key="geo_wd")
    if st.button("Run experimental analysis"):
        try:
            obstacles = []
            if o_pts.strip():
                obstacles.append({"name": o_name or "obstacle", "footprint_px": parse_points(o_pts),
                                  "height_m": [min(o_lo, o_hi), max(o_lo, o_hi)]})
            sc = {"eye_ground_px": [ex, ey], "eye_height_m": [min(eye_lo, eye_hi), max(eye_lo, eye_hi)],
                  "pedestrian_foot_px": [px, py], "pedestrian_height_m": [min(ped_lo, ped_hi), max(ped_lo, ped_hi)],
                  "obstacles": obstacles, "samples": 300, "seed": 1}
            if wi.strip():
                if not obstacles:
                    raise ValueError("a what-if needs an obstacle footprint")
                sc["setback"] = {"obstacle": obstacles[0]["name"], "direction_px": parse_points(wi, 2)[:2],
                                 "distances_m": [float(x) for x in dists.split(",") if x.strip()]}
            with st.spinner("Running sensitivity analysis..."):
                rep = geometry.assess(cal, sc)
            out = OUT / "visibility"
            atomic_write_text(out / "visibility_report.json", json.dumps(rep, indent=2, default=str))
            md = geometry.report_markdown(rep)
            atomic_write_text(out / "visibility_report.md", md)
            st.markdown(md)
            st.caption(f"Saved to {out}")
        except (ValueError, OSError) as e:
            st.error(f"Could not run: {e}")


with tab_video:
    video_tab()
with tab_review:
    review_tab()
with tab_geo:
    geometry_tab()
