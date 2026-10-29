r"""Build a shareable zip of the project WITHOUT your video, outputs, virtual environment or caches.

    python tools\make_package.py            # writes SightLine_package.zip next to the project folder

Excluded on purpose: data\ (except data\README.md), outputs\ (except outputs\.gitkeep), .venv, __pycache__,
config\zones.json and config\calibration.json (your own measurements), any video file, and zip files.
The dataset licence forbids redistributing its videos or frames.
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.paths import project_root  # noqa: E402

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}
KEEP = {"data/README.md", "outputs/.gitkeep"}
SKIP_FILES = {"config/zones.json", "config/calibration.json"}


def included(rel: str) -> bool:
    parts = rel.split("/")
    if rel in KEEP:
        return True
    if parts[0] in {"data", "outputs", ".venv", "outputs_backup"} or "__pycache__" in parts:
        return False
    if rel in SKIP_FILES or Path(rel).suffix.lower() in VIDEO_EXT | {".zip", ".pyc"}:
        return False
    return True


def build(root: Path, dest: Path) -> int:
    n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if p.is_file() and included(rel):
                z.write(p, f"{root.name}/{rel}")
                n += 1
    return n


def main():
    root = project_root()
    dest = root.parent / "SightLine_package.zip"
    print(f"Wrote {dest} with {build(root, dest)} files (no video, outputs or personal config).")


if __name__ == "__main__":
    main()
