# SightLine AI (hackathon prototype)

Explainable pedestrian-visibility **review workflow** for roadside video: find candidate pedestrian-vehicle
moments, look at the evidence, record reviewer judgements with uncertainty, and export them. A gated, experimental
geometry stage exists but produces nothing without a real calibration.

**What this is not:** a safety verdict, a driver-view model, validated on Indian roads, or production-ready.
The only footage used is one 5-minute Bangkok research video (development sample). Read `docs/LIMITATIONS.md`.

## Status by stage

| Stage | What | State |
|---|---|---|
| A | Environment and video audit | Done, run on the target PC and video |
| B | Viewer, seeking, frame export, manual log | Done, run by the user (earlier version) |
| C | Detection, tracking, candidate events | Run by the user on one 60 s window; rules need evaluation |
| D/E | Review UI, evidence sheets, dedup, uncertainty, export | Built; checked against a fake Streamlit only |
| F/G | Gated line-of-sight and labelled what-if | Built and unit-tested on synthetic geometry; **no real calibration, so no results** |
| H | Tests, docs, demo script, packaging | Done; evaluation on real data still to do |
| Audit | Full code audit, event-persistence fix, regression tests | Done in the sandbox; real-Streamlit check is yours: `docs\ACCEPTANCE_TEST.md`, list of fixes: `docs\AUDIT_REPORT.md` |

## Setup (Windows 11, Command Prompt)

```bat
cd path\to\SightLine
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-detect.txt
```
PowerShell: activate with `.\.venv\Scripts\Activate.ps1`. Python 3.11 to 3.13 has been used. Put your video at
`data\video_006.avi` (copy it; it is git-ignored). After a working install run `pip freeze > requirements.lock.txt`.

## Workflow

1. **Audit the video** (once): `python tools\audit_video.py data\video_006.avi --full`
2. **Draw the crosswalk zones** (once, needs a screen): `python tools\define_zones.py data\video_006.avi --frame 1800 --out config\zones.json`
3. **Track a window** (about 13 minutes of CPU per minute of video): `python tools\run_tracking.py data\video_006.avi --start 60 --end 120 --stride 3`
4. **Build candidates**: `python tools\make_candidates.py data\video_006.avi`
   (steps 3 and 4 together: `run_pipeline.bat 60 120`)
5. **Review**: `streamlit run app.py`, tab "Review candidates". Decisions go to `outputs\reviews.json`;
   "Export reviewed events" writes `outputs\reviewed\reviewed_events.csv/json`.
6. **Evaluate**: note reference moments (see `docs\EVALUATION_PLAN.md`), then
   `python tools\evaluate.py data\video_006.avi --gt-csv config\my_moments.csv --window 60 120`
7. **Experimental geometry** (only with real measurements, see `docs\GEOMETRY_GUIDE.md`):
   `python tools\visibility_check.py`

## Where your data lives (and what is safe)

| File | What | Written by |
|---|---|---|
| `outputs\events.json` | **Authoritative** manual events, saved the moment you press Add event | app, `tools\manage_events.py` |
| `outputs\events.csv` | Derived copy of the same events | rewritten after every save |
| `outputs\evidence\manual\` | One evidence image per manual event (never overwritten) | app |
| `outputs\reviews.json` | Your decisions on automated candidates, keyed by candidate | app |
| `outputs\candidates\` | Automated proposals; a re-run moves the old ones to `candidates\backups\` | `make_candidates` |
| `outputs\backups\` | Copies made before any delete, edit, import, migration or unreadable-file event | automatic |
| `outputs\deleted_evidence\` | Evidence images of events you deleted (moved, not erased) | app |

All paths are relative to the project folder, not to where you start Streamlit. Command line access to the manual
log: `python tools\manage_events.py list | check | add | import`.

Debugging a missed moment: `python tools\make_candidates.py data\video_006.avi --explain-frame 2384`.

## How candidates are defined

A candidate is a **(vehicle, pedestrian) pair** in a crosswalk zone: the vehicle overlaps the zone
(`--veh-overlap`, default 0.15), the pedestrian is inside it or within `--ped-margin` pixels (default 120), and
they are within `--max-gap-heights` pedestrian heights of each other (default 3). Everything is in image pixels.
Likely motorcycle riders are excluded only if their motorcycle was also detected. The defaults were tuned on one
hand-labelled moment and are **not validated**. Each candidate carries plain-language caution flags.

## Folder structure

```
SightLine/
  app.py                  Streamlit workbench (3 tabs)
  run_pipeline.bat        tracking + candidates in one go
  requirements.txt        base dependencies
  requirements-detect.txt + Ultralytics, lap (large download)
  config/                 zones.json (yours), calibration/scenario examples
  data/                   your video (git-ignored)
  outputs/                everything generated (git-ignored)
  sightline/              core library (see docs/ARCHITECTURE.md)
  tools/                  command-line stages
  tests/                  unit, smoke and end-to-end tests
  docs/                   architecture, limitations, evaluation plan, demo script, geometry guide, sources
```

## Tests

```bat
python -m unittest discover -s tests -v
```
The suite covers core logic, tools on synthetic data, persistence and failure handling, and the app against a fake
Streamlit (real Streamlit could not be installed where this was built; use `docs\ACCEPTANCE_TEST.md`). It does not run the
detector (no model download in tests) and does not replace manually running the app.

## Licences and data

The sample video is from the Crosswalk dataset: educational, non-commercial use, no redistribution, citation
required. Do not commit or publish the video or frames. Ultralytics is AGPL-3.0. See `docs\SOURCES.md`.
