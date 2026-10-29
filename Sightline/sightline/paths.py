"""One deliberate project root, so nothing depends on the folder you happened to start Python or Streamlit in.

The root is the folder that contains the `sightline` package. Set the environment variable SIGHTLINE_ROOT to use
another folder (the automated tests do this so they never touch your real outputs). Paths stored inside data files
are written relative to the root (with forward slashes) and resolved against it when read, so the whole project
folder can be moved.
"""
from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    env = os.environ.get("SIGHTLINE_ROOT")
    return Path(env).resolve() if env else _PACKAGE_ROOT


def __getattr__(name):                 # PROJECT_ROOT, OUTPUTS, CONFIG, DATA are looked up fresh each time
    root = project_root()
    table = {"PROJECT_ROOT": root, "OUTPUTS": root / "outputs", "CONFIG": root / "config", "DATA": root / "data"}
    if name in table:
        return table[name]
    raise AttributeError(name)


def resolve(p, root: Path | None = None) -> Path:
    """A path as given: absolute stays absolute, relative means relative to the project root (not the cwd)."""
    p = Path(p)
    return p if p.is_absolute() else (root or project_root()) / p


def to_stored(p, root: Path | None = None) -> str:
    """Form used inside JSON/CSV: relative to the root with '/' when inside it, otherwise absolute."""
    if p in (None, ""):
        return ""
    root = root or project_root()
    p = Path(p)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)
