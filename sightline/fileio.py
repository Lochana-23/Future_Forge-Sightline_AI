"""Safe file writing: atomic replace, and backups before anything destructive."""
from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def atomic_write_bytes(path, data: bytes) -> Path:
    """Write to a temporary file in the same folder, then replace. A crash leaves the old file or the new one,
    never half of each. Raises OSError on failure and leaves the destination untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def csv_text(fieldnames, rows) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def atomic_write_csv(path, fieldnames, rows) -> Path:
    """UTF-8 with BOM so Excel opens it cleanly."""
    return atomic_write_bytes(path, csv_text(fieldnames, rows).encode("utf-8-sig"))


def backup_file(path, backups_dir, tag: str = "") -> Path | None:
    """Copy `path` into backups_dir with a timestamp. Returns the copy, or None if there was nothing to copy.
    Raises OSError if the copy cannot be made (callers must then NOT proceed with the destructive step)."""
    path = Path(path)
    if not path.is_file():
        return None
    backups_dir = Path(backups_dir)
    backups_dir.mkdir(parents=True, exist_ok=True)
    dest = backups_dir / f"{path.stem}{'_' + tag if tag else ''}_{stamp()}{path.suffix}"
    shutil.copy2(path, dest)
    return dest
