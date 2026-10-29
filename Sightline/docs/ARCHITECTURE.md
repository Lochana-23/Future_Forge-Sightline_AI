# Architecture

SightLine is a local, file-based pipeline. Every stage reads files and writes files, so any stage can be re-run,
inspected or replaced. Automated outputs and human decisions are always stored separately.

```
video_006.avi ──► tools/run_tracking.py ──► outputs/tracks/tracks.csv (+ preview.avi, summary.json)
                                                   │
config/zones.json ◄── tools/define_zones.py        ▼
        │                                  tools/make_candidates.py
        └──────────────────────────────────►  outputs/candidates/candidates.csv|json (+ evidence images/sheets)
                                                   │
                                                   ▼
                              app.py  "Review candidates" tab  ──►  outputs/reviews.json   (human decisions only)
                                                   │
                                                   ▼
                       outputs/reviewed/reviewed_events.csv|json  (automated columns + human_* columns)
                                                   │
        tools/evaluate.py  ◄── reference moments (your list or dataset annotations)
                                                   │
                       outputs/evaluation/evaluation_report.json

config/calibration.json + config/scenario.json ──► tools/visibility_check.py (or app tab 3) ──► outputs/visibility/
   (gated: refuses without a real calibration)
```

## Modules (`sightline/`)

| Module | Job | Needs a model? |
|---|---|---|
| `video.py` | open video, seek, read frames, export JPEG | no |
| `nav.py`, `uihelp.py` | pure helpers for the app (testable) | no |
| `events.py` | event record + manual log + CSV/JSON export | no |
| `tracks.py` | track rows, CSV I/O, reliability summary | no |
| `candidates.py` | zones, pair-based candidate events, flags, diagnostics | no |
| `evidence.py` | annotated frames and contact sheets | no |
| `review.py` | human review store, flags explained, export | no |
| `evaluate.py` | match candidates to reference moments | no |
| `geometry.py` | gated calibration, line-of-sight, labelled what-if | no |

Only `tools/probe_detect.py` and `tools/run_tracking.py` use the detector (Ultralytics YOLO, CPU).

## Design decisions

- **Streamlit, not FastAPI + React.** One process, no API server; fewer things to break on a student laptop.
- **Files, not a database.** CSV/JSON are inspectable and diff-able. SQLite is a possible later step.
- **Candidates are image-space heuristics.** Everything is in pixels; there are no real-world distances unless the
  gated geometry stage has a real calibration.
- **Human review is separate.** `reviews.json` is keyed by `candidate_key`; automated records are never edited.
- **Contact sheets, not video clips.** Browsers cannot reliably play the MPEG-1 `.avi`, and H.264 encoders are not
  guaranteed on every PC.
- **Geometry refuses by default.** Without 5+ real ground points and a passing error check it produces no numbers.

## Alternatives considered

- Detector: larger YOLO models are more accurate but several times slower on CPU. RT-DETR and others were not tried.
- Tracker: ByteTrack (Ultralytics default). BoT-SORT adds re-identification and may reduce identity switches; untested.
- Geometry: a depth model could estimate scene structure, but its metric accuracy from a single frame is unproven here,
  and it would blur the line between measurement and model estimate.

## Data persistence (added in the audit)

- `sightline/paths.py`: single project root (override with `SIGHTLINE_ROOT`, used by tests).
- `sightline/fileio.py`: atomic writes, timestamped backups.
- `sightline/store.py`: `EventStore` for manual events. JSON is authoritative, CSV derived, permanent `next_id`,
  reload-before-write, backup before destructive changes, read-only when the file is unreadable, evidence image and
  record saved as a pair.
- `sightline/review.py`: `ReviewStore` keyed by candidate key; reviews also store a description of their candidate so
  `reconcile()` can propose (never silently apply) matches after candidates are regenerated.
- Automated candidates, human reviews and manual events are three separate files and are never merged in place.
