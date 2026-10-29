"""Stage E/H: evaluate candidates against reference moments and human review.

Reference moments - either:
  --gt-csv  my_moments.csv       (columns: time_s [or frame], optional end_time_s [or end_frame], note)
  --gt-dir  path\\to\\video_006     (the dataset's annotation folder for this video)

    python tools\\evaluate.py data\\video_006.avi --gt-csv config\\my_moments.csv --window 60 120
    python tools\\evaluate.py data\\video_006.avi --gt-dir annotations\\video_006 --window 60 120

Reports: reference recall (time overlap), plus the human-review summary if reviews exist.
Read the caveats in the report: they matter.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.fileio import atomic_write_text  # noqa: E402
from sightline.evaluate import load_gt_annotations_dir, load_gt_csv, match_events  # noqa: E402
from sightline.paths import resolve  # noqa: E402
from sightline.review import ReviewStore, load_candidates, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--candidates", default="outputs/candidates/candidates.json")
    ap.add_argument("--reviews", default="outputs/reviews.json")
    ap.add_argument("--gt-csv")
    ap.add_argument("--gt-dir")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "END_S"),
                    help="only evaluate this part of the video (the part you actually tracked)")
    ap.add_argument("--tol", type=float, default=2.0, help="seconds of timing tolerance")
    ap.add_argument("--out", default="outputs/evaluation")
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(resolve(a.video)))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        sys.exit("The video reports an invalid frame rate; cannot convert times to frames.")
    try:
        events = load_candidates(resolve(a.candidates))
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read candidates: {e}")

    report = {"video": Path(a.video).name, "fps": fps}
    if a.gt_csv or a.gt_dir:
        try:
            if a.gt_csv:
                gts, stats = load_gt_csv(a.gt_csv, fps), None
            else:
                gts, stats = load_gt_annotations_dir(a.gt_dir)
        except (OSError, ValueError) as e:
            sys.exit(f"Could not read reference moments: {e}")
        report["reference_source"] = a.gt_csv or a.gt_dir
        if stats:
            report["annotation_files"] = stats
        report["matching"] = match_events(events, gts, fps, a.tol, tuple(a.window) if a.window else None)
    else:
        print("No reference moments given (--gt-csv or --gt-dir): skipping recall.")

    try:
        store = ReviewStore(resolve(a.reviews))
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read reviews: {e} (the file was not changed)")
    report["human_review"] = summary(events, store)

    out = resolve(a.out)
    atomic_write_text(out / "evaluation_report.json", json.dumps(report, indent=2))

    print(f"\nCandidates: {len(events)}")
    m = report.get("matching")
    if m:
        print(f"Reference moments in window: {m['reference_moments']}  matched by a candidate: "
              f"{m['reference_matched']}  (recall vs reference: {m['recall_vs_reference']})")
        print(f"Candidates overlapping a reference moment: {m['candidates_matching_a_reference']} of {m['candidates']}")
        for u in m["unmatched_reference"][:10]:
            print(f"  missed reference moment at {u['time_s']}s (class {u['class']})")
    h = report["human_review"]
    print(f"Human review: {h['reviewed']} of {h['candidates']} reviewed; confirmed {h['confirmed']}, "
          f"rejected {h['rejected']}, uncertain {h['uncertain']}; confirmed share of decided: "
          f"{h['confirmed_share_of_decided']}")
    print(f"\nSaved {out / 'evaluation_report.json'}")
    print("Remember: unmatched candidates are not necessarily false alarms, and recall is only as good as the "
          "reference list. These numbers are for ONE video window.")


if __name__ == "__main__":
    main()
