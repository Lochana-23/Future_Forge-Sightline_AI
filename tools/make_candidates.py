"""Stage C3/E: turn tracks + crosswalk zones into candidate events with evidence images.

    python tools\\make_candidates.py data\\video_006.avi --tracks outputs\\tracks\\tracks.csv ^
        --zones config\\zones.json --out outputs\\candidates

Each candidate is a (vehicle, pedestrian) pair in one zone. Writes candidates.csv / candidates.json (same columns
as the manual event log) plus, per candidate, an annotated evidence frame and a multi-frame contact sheet.
Every row is source=auto, review_status=unreviewed. Nothing here is a confirmed event or a visibility judgement.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.candidates import diagnose, explain_frame, find_candidates, load_zones, unique_keys  # noqa: E402
from sightline.fileio import stamp  # noqa: E402
from sightline.paths import resolve, to_stored  # noqa: E402
from sightline.privacy import blur_regions  # noqa: E402
from sightline.events import EventLog  # noqa: E402
from sightline.evidence import choose_sheet_frames, draw_event_boxes, make_sheet  # noqa: E402
from sightline.tracks import read_tracks_csv  # noqa: E402
from sightline.video import VideoError, VideoSource  # noqa: E402

LIMITS = ("Automated candidate from image-space rules: a vehicle track and a pedestrian track are close together "
          "in or near the crosswalk zone. Not verified. Does not show the pedestrian was hidden from, or unseen "
          "by, any driver; tracks can switch identity, miss objects or mislabel classes; pixel gaps are not "
          "real distances.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tracks", default="outputs/tracks/tracks.csv")
    ap.add_argument("--zones", default="config/zones.json")
    ap.add_argument("--out", default="outputs/candidates")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds of gap that split/merge events")
    ap.add_argument("--min-obs", type=int, default=2)
    ap.add_argument("--keep-riders", action="store_true", help="do not exclude likely motorcycle riders")
    ap.add_argument("--far-ped-heights", type=float, default=2.0,
                    help="flag candidates whose closest pedestrian is farther than this many pedestrian heights")
    ap.add_argument("--max-gap-heights", type=float, default=3.0,
                    help="a vehicle and pedestrian only form an event when closer than this many pedestrian heights")
    ap.add_argument("--veh-overlap", type=float, default=0.15,
                    help="min fraction of a vehicle box that must overlap the zone")
    ap.add_argument("--ped-margin", type=float, default=120.0,
                    help="pedestrians within this many pixels of a zone count as near it (0 = strictly inside)")
    ap.add_argument("--explain-frame", type=int, default=None,
                    help="print what was detected near this frame and how it relates to the zones, then exit")
    ap.add_argument("--no-sheets", action="store_true", help="skip contact sheets (faster)")
    ap.add_argument("--blur", action="store_true",
                    help="best-effort pixelation of head and plate areas of DETECTED boxes in evidence images "
                         "(not an anonymisation guarantee; see sightline/privacy.py)")
    a = ap.parse_args()

    try:
        vid = VideoSource(resolve(a.video))
        rows = read_tracks_csv(resolve(a.tracks))
        zones = load_zones(resolve(a.zones))
    except (OSError, ValueError, KeyError, VideoError) as e:
        sys.exit(f"Could not read video/tracks/zones: {e}")
    if not rows:
        sys.exit("tracks file has no rows")
    fps = vid.fps
    riders = not a.keep_riders

    if a.explain_frame is not None:
        print("\n".join(explain_frame(rows, zones, a.explain_frame, exclude_riders=riders, veh_overlap=a.veh_overlap,
                                      ped_margin_px=a.ped_margin, max_gap_ped_heights=a.max_gap_heights)))
        return
    diag = diagnose(rows, zones, exclude_riders=riders, veh_overlap=a.veh_overlap, ped_margin_px=a.ped_margin)
    print(f"Diagnostics ({diag.pop('_processed_frames')} processed frames):")
    for zname, d in diag.items():
        print(f"  {zname}: pedestrian inside in {d['frames_with_pedestrian_inside']} frames "
              f"(inside or near: {d['frames_with_pedestrian_inside_or_near']}), vehicle overlapping in "
              f"{d['frames_with_vehicle']}, both in {d['frames_with_both']}; rider boxes excluded "
              f"{d['rider_boxes_excluded']}; person boxes without a track id skipped "
              f"{d['id_less_person_boxes_skipped']}")
    cands = find_candidates(rows, zones, fps, gap_s=a.gap, min_obs=a.min_obs, exclude_riders=riders,
                            far_ped_heights=a.far_ped_heights, veh_overlap=a.veh_overlap,
                            ped_margin_px=a.ped_margin, max_gap_ped_heights=a.max_gap_heights)

    out = resolve(a.out)
    ev_dir = out / "evidence"
    # Never overwrite a previous run silently: keep its files (and evidence images) in out/backups/<timestamp>/.
    old = [p for p in (out / "candidates.json", out / "candidates.csv", ev_dir) if p.exists()]
    if old:
        bdir = out / "backups" / stamp()
        bdir.mkdir(parents=True, exist_ok=True)
        for p in old:
            shutil.move(str(p), str(bdir / p.name))
        print(f"Previous candidates and evidence moved to {bdir} (your reviews are stored separately and untouched).")
    ev_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r.frame, []).append(r)

    log = EventLog()
    keys = unique_keys(cands)
    for i, c in enumerate(cands, start=1):
        vb, pb = ",".join(f"{v:.0f}" for v in c.vehicle_box), ",".join(f"{v:.0f}" for v in c.ped_box)
        poly = zones[c.zone]
        ev_path = sheet_path = ""
        try:
            frames = [c.evidence_frame]
            if not a.no_sheets:
                frames = choose_sheet_frames(by_frame.keys(), c.start_frame, c.end_frame, c.evidence_frame, fps)
            imgs, labs = [], []
            for f, frame in vid.iter_frames(frames):
                if frame is None:
                    continue
                if a.blur:
                    frame = blur_regions(frame, by_frame.get(f, []))
                img = draw_event_boxes(frame, poly, c.vehicle_tracks, c.pedestrian_tracks, by_frame.get(f, []))
                if f == c.evidence_frame:
                    ev_path = str(ev_dir / f"cand_{i:03d}_f{f}.jpg")
                    if not cv2.imwrite(ev_path, img):
                        raise OSError(f"could not write {ev_path}")
                imgs.append(img)
                labs.append(f"frame {f} | {f / fps:.1f}s" + ("  <- evidence" if f == c.evidence_frame else ""))
            if not a.no_sheets and imgs:
                sheet_path = str(ev_dir / f"cand_{i:03d}_sheet.jpg")
                if not cv2.imwrite(sheet_path, make_sheet(imgs, labs)):
                    raise OSError(f"could not write {sheet_path}")
        except Exception as e:  # evidence is nice-to-have; never lose the candidate itself
            print(f"  warning: evidence for candidate {i} failed: {e}")
            ev_path = sheet_path = ""
        flags = list(c.flags)
        if c.riders_excluded:
            flags.append(f"{c.riders_excluded}_likely_rider_boxes_excluded_in_zone")
        log.add(c.evidence_frame, fps, source="auto", review_status="unreviewed",
                event_type="vehicle and pedestrian close together at crosswalk (candidate)",
                model_confidence=f"{c.det_conf_mean:.2f} (mean detector score; not correctness)",
                notes=f"start {c.start_frame / fps:.1f}s end {c.end_frame / fps:.1f}s; {c.n_obs} observations; "
                      f"closest pixel gap {c.min_px_gap}px = {c.gap_ped_heights} pedestrian heights "
                      f"(image-space, not metres)",
                limitations=LIMITS, evidence_frame=to_stored(ev_path), evidence_sheet=to_stored(sheet_path),
                video=Path(a.video).name, frame_start=c.start_frame,
                frame_end=c.end_frame, zone=c.zone, vehicle_class=c.vehicle_class,
                vehicle_tracks=";".join(map(str, c.vehicle_tracks)),
                pedestrian_tracks=";".join(map(str, c.pedestrian_tracks)), flags=";".join(flags),
                candidate_key=keys[i - 1], vehicle_box=vb, pedestrian_box=pb)
    log.to_csv(out / "candidates.csv")
    log.to_json(out / "candidates.json", Path(a.video).name)

    n_flag = sum(1 for e in log.events if e.flags)
    print(f"\n{len(cands)} candidate events from {len(rows)} detections ({n_flag} carry caution flags).")
    for e in log.events[:20]:
        print(f"  #{e.event_id} {e.time_s:6.1f}s zone={e.zone} veh={e.vehicle_class}[{e.vehicle_tracks}] "
              f"ped=[{e.pedestrian_tracks}] flags={e.flags or '-'}")
    if len(log.events) > 20:
        print(f"  ... {len(log.events) - 20} more in candidates.csv")
    n_ov = sum(1 for e in log.events if "overlaps_candidate" in e.flags)
    if n_ov:
        print(f"{n_ov} candidates overlap another one in time and share a track id (flagged): they may be one "
              f"prolonged interaction split by the pairing rule; review them together.")
    print(f"\nSaved candidates.csv, candidates.json and evidence images in {out}")
    if not a.blur:
        print("Evidence images are NOT anonymised (faces/plates may be visible). Do not publish them; "
              "use --blur for best-effort pixelation.")
    print("Candidates are NOT confirmed events. Review every one (streamlit run app.py -> Review tab).")


if __name__ == "__main__":
    main()
