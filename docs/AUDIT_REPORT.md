# Audit report: bugs found and fixes

Every item below was found by reading the code and, for the first one, reproducing it. "Test" names the automated
test that now guards it. Nothing here was verified in the real Streamlit (see "Not verified").

## Critical

| # | Bug | Cause | Fix | Test |
|---|---|---|---|---|
| 1 | Earlier manual events vanish; new export has id 1 | Events lived only in `st.session_state`; a new session started an empty `EventLog`, and Export rewrote both files from it | New `sightline/store.py` (`EventStore`): the event is written to `outputs/events.json` when you press Add event, memory is reloaded from disk before every change, `events.csv` is derived from the JSON | `test_store.py` (whole file), `TestAppPersistenceFlows` |
| 2 | Ids restart or are reused | `max(id)+1` from the in-memory list | Permanent `next_id` stored in the file; deleting the newest event does not free its id | `test_delete_persists_backs_up_and_never_reuses_the_id` |
| 3 | A half-written or failed save could be shown as success | Plain `write_text`, no error handling around export | Atomic write (temp file, then replace); a failed JSON write raises and changes neither disk nor memory; a failed CSV write is reported separately | `test_failed_json_write_...`, `test_csv_failure_...`, `test_unwritable_location_...` |
| 4 | Evidence image and event record could disagree | Image written first, record later, no rollback; shared file name per frame could overwrite another event's image | Image + record saved as a pair with rollback; image named `event_<id>_f<frame>.jpg` and never overwrites an existing file | `TestEvidencePairing` |
| 5 | Corrupt/incompatible events file could be replaced by an empty log | No validation on load | Unreadable file is copied to `outputs/backups`, the store becomes read-only and says why; recovery is an explicit "move it aside" action; the file is moved, never deleted | `test_malformed_json_...`, `test_incompatible_schemas_...` |

## Other defects

| # | Bug | Fix | Test |
|---|---|---|---|
| 6 | Failed add (or any error) cleared the form (`clear_on_submit`) | Form keeps input; notes are cleared only after a successful save | `test_failed_save_is_reported_and_keeps_...` |
| 7 | "Event id to delete" silently did nothing for a missing id; no confirmation; image left behind | Selectbox of existing ids, confirmation tick, backup first, image moved to `outputs/deleted_evidence` | `test_delete_requires_confirmation_then_persists` |
| 8 | Paths depended on the folder Streamlit was started in | `sightline/paths.py` project root; paths stored relative to it; all tools and the app use it | `TestPathsAndPrivacy`, `test_stored_paths_are_relative_...` |
| 9 | Changing the video path left stale frame/slider values (could exceed the new video's range) | Positions reset when the video path changes | `test_switching_video_resets_stale_positions` |
| 10 | One shared `cv2.VideoCapture` used by all browser sessions without a lock | `RLock` around seek+read | (covered by existing video tests; race itself not reproducible in a unit test) |
| 11 | Review store: stale session could overwrite newer decisions; malformed shapes raised uncaught errors; no backup | Reload before save (`decide`), shape validation as `ValueError`, previous version kept in `outputs/backups/reviews_previous.json` | `TestReviewPersistence` |
| 12 | Regenerating candidates silently orphaned reviews (key includes start frame) | Reviews now store a candidate description; `reconcile()` proposes unambiguous matches, the app asks before re-attaching, ambiguous or undescribed ones are only reported | `TestRegeneration` |
| 13 | `make_candidates` overwrote the previous candidates and left stale evidence files | Previous run moved to `candidates/backups/<time>/` | `test_second_run_keeps_the_first_runs_files...` |
| 14 | Pedestrian detections without a track id (id -1) were all treated as one person | Skipped for candidates and counted in the diagnostics | `test_pedestrians_without_track_id_...` |
| 15 | One long interaction could appear as several unexplained candidates | Overlapping candidates in the same zone that share a vehicle or pedestrian id get a plain-language flag (not merged automatically) | `test_overlapping_candidates_...` |
| 16 | Candidate keys could collide within a run | `unique_keys()` | `test_keys_are_unique_within_a_run` |
| 17 | Geometry gate: NaN/inf values could pass (NaN comparisons are false); horizon points gave nonsense; scenario inputs unvalidated; unbounded what-if distances | Finite checks everywhere, horizon check, image_size bounds, scenario validation, sample and distance limits | `TestGeometryHardening` |
| 18 | Geometry tab: prefilled pedestrian position did not update when another candidate was chosen (keyed widgets ignore a changed default) | Selection callback sets the widget values | (UI only; see manual test) |
| 19 | `define_zones.py` overwrote `config/zones.json` without a backup and allowed duplicate zone names | Backup first; duplicate names refused | not automated (needs a screen) |
| 20 | `run_tracking.py` overwrote earlier tracks and lost everything on Ctrl-C | Backup of previous files; Ctrl-C saves what was processed and records `interrupted` | not automated (needs the detector) |
| 21 | Track summary invited over-reading ("0 untracked") | Summary now says what it does not evaluate and counts same-frame duplicate boxes | `TestTracksSummary` |
| 22 | `evaluate.py` crashed with a traceback on a bad reviews file or invalid fps | Clear exit messages | `test_bad_inputs_fail_cleanly` (partly) |
| 23 | Exports (reviewed CSV/JSON, reports) were not atomic | `fileio` atomic writers | `TestReviewPersistence` |
| 24 | Evidence frames show faces/plates with no warning or protection | Warning in the app and tools; optional `--blur` for `make_candidates` (best effort only) | `test_blur_changes_only_...` |

## Not verified (cannot be, here)

- The real Streamlit app: Streamlit cannot be installed in the sandbox used for this work, so the app was run only
  against a fake Streamlit (`tests/test_app_smoke.py`). Use `docs/ACCEPTANCE_TEST.md` on your PC.
- The YOLO tracking, `define_zones.py` window, and the dataset annotation parser on real files.
- Your real `outputs` files: they were not available. The frame-2384 event was reproduced only as a clearly
  labelled test fixture in a temporary folder. If your real `outputs/events.json` no longer contains it, it has to
  be re-entered (`python tools\manage_events.py add ...`) or restored from a backup you may have.
- Candidate defaults remain tuned on one moment; nothing about detector or rule accuracy has been evaluated.
