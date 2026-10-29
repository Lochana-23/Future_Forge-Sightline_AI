r"""Stages F/G (EXPERIMENTAL): gated line-of-sight check and labelled what-if.

    python tools\visibility_check.py --calibration config\calibration.json --scenario config\scenario.json

Needs a REAL calibration (>= 5 ground points measured outside the video, e.g. from satellite imagery).
Without one the tool refuses and prints why; it never invents numbers.
Writes outputs/visibility/visibility_report.json and .md. Exit code 2 if not assessable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.fileio import atomic_write_text  # noqa: E402
from sightline.paths import resolve  # noqa: E402
from sightline.geometry import assess, load_json, report_markdown  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="config/calibration.json")
    ap.add_argument("--scenario", default="config/scenario.json")
    ap.add_argument("--out", default="outputs/visibility")
    a = ap.parse_args()
    try:
        cal, sc = load_json(resolve(a.calibration)), load_json(resolve(a.scenario))
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read inputs: {e}\nCopy config\\calibration.example.json to config\\calibration.json and "
                 f"fill in REAL measurements (see docs\\GEOMETRY_GUIDE.md).")
    try:
        rep = assess(cal, sc)
    except ValueError as e:
        sys.exit(f"Scenario problem: {e}")
    out = resolve(a.out)
    atomic_write_text(out / "visibility_report.json", json.dumps(rep, indent=2, default=str))
    md = report_markdown(rep)
    atomic_write_text(out / "visibility_report.md", md)
    print(md)
    print(f"Saved {out / 'visibility_report.json'} and .md")
    sys.exit(0 if rep["status"] == "EXPERIMENTAL_OK" else 2)


if __name__ == "__main__":
    main()
