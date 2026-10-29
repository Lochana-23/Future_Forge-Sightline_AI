# Demo script (about 5 minutes) and claims checklist

Do not skip the limitation moments: they are part of the demo.

## Before you present

- Run the pipeline on your chosen window so candidates and (some) reviews exist.
- Check the dataset licence: show the live app locally; do not post frames, screenshots or the video publicly.
- Have one good case and one bad case ready (a false alarm or a missed moment).

## Script

1. **Problem (30 s).** "Reviewers of roadside footage need to find pedestrian-vehicle moments and judge what might
   have been hiding the pedestrian. We built a tool for the review workflow, not a safety verdict."
2. **Data honesty (20 s).** "Sample: one 5-minute research video from a Bangkok intersection, elevated fixed
   camera, used only for development. It does not represent Indian roads."
3. **Tab 1, video (40 s).** Seek, step by seconds and frames, log a manual event. "Automated and human records are
   separate."
4. **Pipeline (40 s).** Show `preview.avi` briefly: boxes and track ids. Point at one wrong class or split id.
   "Detections are candidates, not facts."
5. **Tab 2, review (90 s).** Open a candidate: contact sheet, flags in plain words. Confirm one, reject one, mark one
   uncertain. Choose Partial with an obstruction. "This label is my judgement from the camera, not the driver's seat."
6. **Numbers (30 s).** Show the evaluation report: recall vs reference, confirmed share, N and window stated.
7. **Tab 3, geometry (30 s).** "Geometry is experimental and gated. Without real calibration it refuses." Show the
   refusal. If you do have a real calibration, show the Uncertain-able result and the HYPOTHETICAL what-if with its
   "cannot predict" list.
8. **Limits and next steps (30 s).** Single video, tuned thresholds, missed scooters, no validation. Next: more
   footage (including local), second reviewer, calibration, blur for privacy.

## Claims checklist (check your slides and your speech against this)

| Do not say | Say instead |
|---|---|
| "First of its kind" / "novel" without a literature check | "A workflow prototype that combines existing detection/tracking with reviewer-defined visibility labels" |
| "Validated on Indian roads" | "Developed on one Bangkok sample; no Indian data yet" |
| "Production-ready" / "deployable" | "Hackathon prototype for workflow feasibility" |
| "Reduces accidents" / "improves road safety" | "Could help reviewers study situations; no safety effect has been measured" |
| "Detects when the driver could not see the pedestrian" | "Shows reviewer-labelled possible obstructions in the camera view" |
| "Minimum safe set-back distance" | "A hypothetical what-if under stated assumptions; cannot predict behaviour or crashes" |
| "AI accuracy of X%" | Report recall vs reference and review outcomes, with N, window and caveats |
| Showing dataset frames publicly | Show locally, cite the dataset, or use your own footage |

## Questions to expect

- *Why not just use the dataset's labels?* They mark yielding behaviour, not visibility, and they are reference
  material for evaluation only.
- *Is the geometry accurate?* No calibration has been supplied, so no geometry results are claimed.
- *What if the tracker fails?* Every candidate is human-reviewed; flags show likely tracker problems.
