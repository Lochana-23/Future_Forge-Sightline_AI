# Sources, licences and attribution

## Sample video

- Source: the **Crosswalk** dataset (Nanasaki-Ai/Crosswalk on GitHub). Videos 001-040 were downloaded from the
  figshare link in that repository's README.
- Stated terms (from the repository README as read on 2026-10-23; re-read it yourself before relying on this):
  educational use only; no commercial use; redistribution of the dataset, in whole or in part, prohibited; users
  must cite the original source; no warranty; follow applicable laws and research ethics; the owning institution
  keeps the final right of interpretation.
- The README does not name a standard licence, and says nothing about consent from people filmed or about blurring
  faces and number plates.
- Citation: use the BibTeX given in the dataset README (a TITS 2025 paper by Hu et al.). Copy it from there rather
  than from this file.
- What this means here: do not commit the video or frames to a public repository, do not post screenshots, show the
  tool locally. `data/` and `outputs/` are git-ignored for this reason.

## Software (check each licence yourself before distributing)

| Component | Licence (as understood) | Note |
|---|---|---|
| Ultralytics YOLO | AGPL-3.0 | Copyleft: fine for a local prototype, check before distributing or hosting |
| OpenCV | Apache-2.0 | |
| Streamlit | Apache-2.0 | |
| NumPy, pandas | BSD-3-Clause | |

## Work that is ours versus existing

- Existing: object detection and tracking (YOLO, ByteTrack), the dataset and its yielding annotations.
- Ours (this prototype): the candidate-event rules, the reviewer workflow with separated automated/human records,
  the evaluation helpers, and the gated geometry framework. None of it is validated.
