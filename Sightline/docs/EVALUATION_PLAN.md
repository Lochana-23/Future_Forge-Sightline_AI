# Evaluation plan

Goal: report honestly how well the workflow finds real vehicle-pedestrian moments and how often it raises false
alarms, on the video window you actually processed. This is a feasibility check, not a benchmark.

## What is measured

| Question | Measure | How |
|---|---|---|
| Does it find real moments? | Recall vs a reference list | `tools\evaluate.py --gt-csv` (your list) or `--gt-dir` (dataset annotations) |
| How many candidates are real? | Confirmed share of decided (confirmed / (confirmed + rejected)) | Human review in the app |
| How reliable are tracks? | Short tracks, class flips, merged ids | `outputs\tracks\summary.json`, candidate flags |
| How consistent are labels? | Agreement between two reviewers | Not implemented: have a teammate review the same candidates and compare by hand |

## Protocol

1. Fix the window (for example 60 to 120 s) and the settings (stride 3, default candidate parameters). Write them down.
2. Build the reference list **before** looking at candidates: watch `preview.avi` and note every moment where a vehicle
   and a pedestrian share or approach a crosswalk (`time_s,note` CSV). Or use the dataset's annotation folder for
   this video if it exists (class 0/1 is about yielding, not visibility).
3. Run `make_candidates.py`, then review **every** candidate in the app.
4. Run `evaluate.py`; keep `evaluation_report.json`.
5. Log failures: for each missed reference moment, run `make_candidates.py --explain-frame N` and note the cause
   (missed detection, tracker split, zone shape, thresholds).
6. Only then change parameters, and re-evaluate on a **different** window to avoid tuning on the test data.

## Reporting template (copy into the demo)

- Video, window, settings, date.
- Reference moments: N (source). Matched: M. Recall vs reference: M/N.
- Candidates: C. Reviewed: R. Confirmed: a, rejected: b, uncertain: c.
- Three failure cases with screenshots (kept private if the dataset licence applies).
- Statement: "Workflow feasibility on one 60 s window of one Bangkok video. Not a validation."

## Caveats you must state

- Recall is only meaningful if the reference list is reasonably complete.
- An unmatched candidate is not automatically a false alarm; the review decision is the false-alarm measure.
- Matching is by time only.
- Numbers from one window say nothing about other times of day, weather or places.
- Tuning parameters on the same window used for scoring inflates results.
