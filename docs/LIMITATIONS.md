# Limitations and risk register

Ranked by severity (S), likelihood (L) and impact on a reviewer or audience (I): H = high, M = medium, L = low.
"Status" says what is actually in place, not what is hoped for.

| # | Issue | S | L | I | Mitigation in this project | Status |
|---|---|---|---|---|---|---|
| 1 | Image overlap or geometric visibility read as "the driver could not see" | H | M | H | Camera view is stated everywhere; labels are reviewer-defined; geometry is gated, experimental and labelled | In place; depends on honest presentation |
| 2 | Geometry from uncalibrated or weakly calibrated pixels | H | H | H | Gate refuses without >= 5 real ground points and a passing leave-one-out error; never invents distances | In place; **no calibration exists yet, so no geometry results exist** |
| 3 | What-if read as a prediction of behaviour or crash reduction | H | M | H | Output always carries "HYPOTHETICAL", lists what is held constant and what cannot be predicted | In place |
| 4 | Missed detections (small, occluded, edge-of-frame people and scooters) | M | H | H | Human review of every candidate; reference-moment recall check | Partly: recall not yet measured on real data |
| 5 | Identity switches and fragmented tracks (one person, many ids) | M | H | M | Handover merging, flags (`merged_*`), ids shown in review | Partial; merging thresholds unvalidated |
| 6 | Class confusion (car/bus/truck flips; riders as pedestrians) | M | H | M | `vehicle_class_flip` flag; rider boxes excluded only if the motorcycle was detected | Partial; riders of undetected scooters still appear as pedestrians |
| 7 | Candidate rules tuned on one hand-labelled moment | M | H | M | Defaults documented as unvalidated; parameters exposed; evaluation tool | Open until evaluated on a larger reference list |
| 8 | Dataset licence (educational, non-commercial, no redistribution) | H | M | H | `data/` and `outputs/` git-ignored; docs say do not publish frames; cite the source | In place; check before any public demo or prize entry |
| 9 | Privacy: faces and number plates appear in frames and evidence images | M | M | M | Evidence stays local; no uploads; do not publish exports | No blurring implemented (optional next step) |
| 10 | Bangkok footage presented as Indian road evidence | H | L | H | Stated in the app, README and demo script | In place; wording discipline needed |
| 11 | Camera motion, lighting and weather changes | M | L | M | Audit found no large frame jumps in this clip | Only one clip examined |
| 12 | Single video, single window (60-120 s) | M | H | H | Reported as workflow feasibility only | Open |
| 13 | Dependency drift (OpenCV 5 removed HOG; Streamlit API changes) | M | M | M | Smoke test with a fake Streamlit; `pip freeze` lock recommended | Partial: real Streamlit not tested in CI |
| 14 | Reviewer bias and inconsistency | M | M | M | Label guide in the app; "Uncertain" encouraged | No second reviewer / agreement measure yet |
| 15 | Reproducibility | M | M | M | Seeded sampling, file-based pipeline, tests | Dependency versions are not locked yet |
| 17 | Evidence images show faces and plates | M | M | M | Warnings in app and tools; optional best-effort `--blur` (head and plate areas of DETECTED boxes only) | Manual-event evidence frames are never blurred; blur is not a guarantee |
| 18 | Two people editing the same outputs at once | L | L | M | Reload-before-write in both stores; no cross-process lock | Last writer wins only for the same record |
| 16 | Ultralytics is AGPL-3.0 | M | L | M | Documented in `requirements-detect.txt` and `SOURCES.md` | Check before distributing beyond a prototype |

## What the prototype can and cannot show

Can show: that a reviewer can load a video, step through automatically proposed pedestrian-vehicle moments, see
evidence, record decisions with uncertainty, and export the result, with the limits visible at every step.

Cannot show: how often drivers fail to see pedestrians, that any obstruction causes a crash, that a set-back
would help, validity on Indian roads, or production readiness.

## Known gaps

- Detector and tracker run only on a chosen window (about 13 minutes of CPU time per minute of video at stride 3).
- No second-reviewer agreement measure.
- No face or plate blurring.
- Only an earlier version of the video tab has been run in real Streamlit by the user. The current app was checked
  only against a fake Streamlit (see `tests/test_app_smoke.py`); run `docs/ACCEPTANCE_TEST.md`.
- Tracking quality (identity switches, missed objects, class accuracy) has not been measured; the track summary
  lists what it does not evaluate. Candidate rules are unvalidated.
- Reviews can be re-attached after candidates are regenerated only when the same tracks.csv ids are reused; after
  re-tracking, old reviews are reported as unmatched, never guessed.
