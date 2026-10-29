"""Persistent store for MANUAL (human-entered) events.

Design (read this before changing anything):
  * `outputs/events.json` is the ONE authoritative record. `outputs/events.csv` is derived from it and is rewritten
    after every successful JSON save, so the two cannot diverge except when the CSV write itself fails (reported).
  * Every change (add / edit / delete / import) is: reload from disk -> validate -> write JSON atomically -> only then
    update memory. If the write fails, memory and disk are unchanged and an EventStoreError says why.
  * `next_id` is stored in the file and only ever increases, so deleting the newest event never lets its id be
    reused, and ids stay unique across restarts.
  * Before anything destructive (delete, edit, import, schema migration) the current file is copied to
    `outputs/backups/`. A file that cannot be read is NEVER replaced: the store goes read-only (`blocked`) and
    explains; you must explicitly quarantine it (it is moved, not deleted) to start a new log.
  * An evidence image and its event record are saved as a pair: if the record cannot be saved, the image just
    written is removed again; if the image cannot be written, no record is created.
Automated candidates are NOT stored here; they live in outputs/candidates and human reviews in outputs/reviews.json.
"""
from __future__ import annotations

import csv
import json
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .events import FIELD_NAMES, VISIBILITY_LABELS, Event, event_from_row
from .fileio import atomic_write_csv, atomic_write_text, backup_file, stamp
from .paths import to_stored

SCHEMA_VERSION = 2
EDITABLE = ("event_type", "visibility_label", "notes")
_LOCK = threading.RLock()


class EventStoreError(Exception):
    """A save/load problem with a message meant to be shown to the user."""


@dataclass
class SaveResult:
    json_path: Path
    csv_path: Path
    csv_error: str | None = None      # set when the JSON was saved but the derived CSV could not be written
    backup: Path | None = None
    notes: tuple = ()


class EventStore:
    def __init__(self, outputs_dir):
        self.dir = Path(outputs_dir)
        self.root = self.dir.parent                      # stored paths are relative to this
        self.json_path = self.dir / "events.json"
        self.csv_path = self.dir / "events.csv"
        self.backups = self.dir / "backups"
        self.evidence_dir = self.dir / "evidence" / "manual"
        self.deleted_dir = self.dir / "deleted_evidence"
        self.events: list[Event] = []
        self.next_id = 1
        self.blocked: str | None = None                  # set when the file cannot be read: store is read-only
        self.needs_migration = False
        self.csv_error: str | None = None
        self.load_notes: list[str] = []
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self):
        self.events, self.next_id, self.blocked = [], 1, None
        self.needs_migration, self.load_notes = False, []
        if not self.json_path.exists():
            if self.csv_path.exists():
                self.load_notes.append(f"{self.csv_path.name} exists but {self.json_path.name} does not. The CSV is "
                                       "not read automatically; use 'Import events from a file' if you want it.")
            return
        try:
            raw = self.json_path.read_text(encoding="utf-8-sig")
            doc = json.loads(raw)
            if not isinstance(doc, dict) or not isinstance(doc.get("events"), list):
                raise ValueError("expected an object with an 'events' list")
            schema = doc.get("schema_version", 1)
            if not isinstance(schema, int) or schema > SCHEMA_VERSION:
                raise ValueError(f"schema_version {schema!r} was written by a newer version of this program")
            events, seen = [], set()
            for i, row in enumerate(doc["events"], start=1):
                ev = event_from_row(row, f"event #{i}")
                if ev.event_id in seen:
                    raise ValueError(f"duplicate event_id {ev.event_id}")
                seen.add(ev.event_id)
                if not ev.video and isinstance(doc.get("video"), str):
                    ev.video = doc["video"]              # legacy files kept the video name at the top level
                events.append(ev)
            stored_next = doc.get("next_id", 0)
            stored_next = stored_next if isinstance(stored_next, int) and not isinstance(stored_next, bool) else 0
            self.next_id = max([stored_next] + [e.event_id + 1 for e in events] + [1])
            self.events = events
            self.needs_migration = schema < SCHEMA_VERSION
            if self.needs_migration:
                self.load_notes.append("Older events file format: it will be upgraded on the next save, after a "
                                       "backup copy is made.")
        except (OSError, ValueError) as e:
            copy = None
            try:
                data = self.json_path.read_bytes()
                same = [b for b in self.backups.glob("events_unreadable_*.json") if b.read_bytes() == data] \
                    if self.backups.is_dir() else []
                copy = same[0] if same else backup_file(self.json_path, self.backups, "unreadable")   # once per content
            except OSError:
                pass
            self.blocked = (f"{self.json_path} could not be read ({e}). It has NOT been changed or replaced"
                            + (f"; a copy was saved as {copy}." if copy else ".")
                            + " Fix or restore the file, or choose 'Move it aside and start a new log'.")

    def reload(self):
        with _LOCK:
            self._load()
        return self

    def _writable(self):
        if self.blocked:
            raise EventStoreError(self.blocked)

    # ------------------------------------------------------------------ writing
    def _doc(self, events, next_id) -> dict:
        videos = [e.video for e in events if e.video]
        return {"schema_version": SCHEMA_VERSION, "video": videos[-1] if videos else "",
                "note": "Manual reviewer events. Labels are reviewer-defined, not ground truth. "
                        "events.csv is derived from this file.",
                "next_id": next_id, "events": [asdict(e) for e in events]}

    def _commit(self, events, next_id, backup_tag: str | None = None) -> SaveResult:
        """Write JSON (atomic), then CSV. Memory is updated only after the JSON write succeeded."""
        backup = None
        try:
            if backup_tag or self.needs_migration:
                backup = backup_file(self.json_path, self.backups, backup_tag or "premigration")
            atomic_write_text(self.json_path, json.dumps(self._doc(events, next_id), indent=2))
        except OSError as e:
            raise EventStoreError(f"Could not save {self.json_path}: {e}. Nothing was changed.") from e
        self.events, self.next_id, self.needs_migration = events, next_id, False
        res = SaveResult(self.json_path, self.csv_path, None, backup)
        try:
            atomic_write_csv(self.csv_path, FIELD_NAMES, [asdict(e) for e in events])
            self.csv_error = None
        except OSError as e:
            self.csv_error = res.csv_error = (f"events.json was saved, but events.csv could not be written ({e}). "
                                              "The JSON is the authoritative copy; use 'Export' to rewrite the CSV.")
        return res

    def add(self, frame: int, fps: float, video: str = "",
            evidence_writer: Callable[[Path], None] | None = None, **fields) -> tuple[Event, SaveResult]:
        """Create one manual event. `evidence_writer(dest_path)` must write the image to dest_path or raise."""
        if fps <= 0:
            raise EventStoreError("fps must be positive")
        if int(frame) < 0:
            raise EventStoreError("frame must be >= 0")
        with _LOCK:
            self._load()
            self._writable()
            eid = self.next_id
            ev_path = None
            try:
                ev = Event(event_id=eid, frame=int(frame), time_s=round(int(frame) / fps, 3), video=video, **fields)
                if ev.visibility_label not in VISIBILITY_LABELS:
                    raise ValueError(f"Invalid visibility label: {ev.visibility_label}")
                if ev.source != "manual":
                    raise ValueError("only manual events belong in this store")
            except (TypeError, ValueError) as e:
                raise EventStoreError(f"Event not saved: {e}") from e
            if evidence_writer is not None:
                ev_path = self._unique_evidence_path(eid, int(frame))
                try:
                    ev_path.parent.mkdir(parents=True, exist_ok=True)
                    evidence_writer(ev_path)
                    if not ev_path.is_file() or ev_path.stat().st_size == 0:
                        raise OSError("the image file was not created")
                except Exception as e:
                    self._remove_quietly(ev_path)
                    raise EventStoreError(f"Evidence image could not be saved ({e}). No event was recorded.") from e
                ev.evidence_frame = to_stored(ev_path, self.root)
            try:
                res = self._commit(self.events + [ev], eid + 1)
            except EventStoreError:
                if ev_path is not None:
                    self._remove_quietly(ev_path)            # do not leave an image that has no record
                raise
            return ev, res

    def update(self, event_id: int, **changes) -> SaveResult:
        bad = [k for k in changes if k not in EDITABLE]
        if bad:
            raise EventStoreError(f"These fields cannot be edited: {bad}. Editable: {list(EDITABLE)}")
        if "visibility_label" in changes and changes["visibility_label"] not in VISIBILITY_LABELS:
            raise EventStoreError(f"Invalid visibility label: {changes['visibility_label']}")
        with _LOCK:
            self._load()
            self._writable()
            if event_id not in {e.event_id for e in self.events}:
                raise EventStoreError(f"No event with id {event_id}.")
            new = [Event(**{**asdict(e), **changes}) if e.event_id == event_id else e for e in self.events]
            return self._commit(new, self.next_id, "preedit")

    def delete(self, event_id: int) -> SaveResult:
        """Remove one event. Its evidence image is MOVED to outputs/deleted_evidence (not erased) unless another
        event still uses it. The id is never reused."""
        with _LOCK:
            self._load()
            self._writable()
            target = next((e for e in self.events if e.event_id == event_id), None)
            if target is None:
                raise EventStoreError(f"No event with id {event_id}; nothing was deleted.")
            rest = [e for e in self.events if e.event_id != event_id]
            res = self._commit(rest, self.next_id, "predelete")
            notes = []
            img = self._own_evidence_file(target)
            if img is not None and not any(e.evidence_frame == target.evidence_frame for e in rest):
                try:
                    self.deleted_dir.mkdir(parents=True, exist_ok=True)
                    dest = self.deleted_dir / f"{stamp()}_{img.name}"
                    shutil.move(str(img), str(dest))
                    notes.append(f"evidence image moved to {dest}")
                except OSError as e:
                    notes.append(f"event deleted, but its evidence image could not be moved ({e})")
            res.notes = tuple(notes)
            return res

    def import_file(self, path) -> tuple[list[Event], int, SaveResult | None]:
        """Add events from another events JSON (any schema). Events that already exist (same video, frame, type,
        label and notes) are skipped; the rest get NEW ids. Returns (added, skipped, result)."""
        from .events import EventLog
        try:
            incoming = EventLog.from_json(path).events
        except (OSError, ValueError) as e:
            raise EventStoreError(f"Cannot import {path}: {e}") from e
        with _LOCK:
            self._load()
            self._writable()

            def sig(e):
                return (e.video, e.frame, e.event_type, e.visibility_label, e.notes)
            have = {sig(e) for e in self.events}
            added, nid, skipped = [], self.next_id, 0
            for e in incoming:
                if sig(e) in have:
                    skipped += 1
                    continue
                have.add(sig(e))
                added.append(Event(**{**asdict(e), "event_id": nid, "source": "manual"}))
                nid += 1
            if not added:
                return [], skipped, None
            return added, skipped, self._commit(self.events + added, nid, "preimport")

    def refresh_exports(self) -> SaveResult:
        """Rewrite events.json and events.csv from the authoritative data. Never changes or drops an event."""
        with _LOCK:
            self._load()
            self._writable()
            return self._commit(list(self.events), self.next_id)

    def quarantine_unreadable(self) -> Path:
        """Explicit recovery: move the unreadable file into backups/ (it is kept) so a new log can start."""
        with _LOCK:
            if not self.blocked:
                raise EventStoreError("The events file is readable; nothing to move aside.")
            self.backups.mkdir(parents=True, exist_ok=True)
            dest = self.backups / f"events_unreadable_moved_{stamp()}.json"
            try:
                shutil.move(str(self.json_path), str(dest))
            except OSError as e:
                raise EventStoreError(f"Could not move the file aside: {e}") from e
            self._load()
            return dest

    # ------------------------------------------------------------------ checks
    def consistency(self) -> dict:
        """Compare events.json with events.csv (id, frame, time, label). ok is False if either is missing/different."""
        out = {"ok": False, "json_events": len(self.events), "csv_events": None, "problems": []}
        if not self.csv_path.is_file():
            out["problems"].append("events.csv does not exist")
            return out
        try:
            with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error) as e:
            out["problems"].append(f"events.csv unreadable: {e}")
            return out
        out["csv_events"] = len(rows)
        by_id = {}
        for r in rows:
            try:
                by_id[int(r["event_id"])] = r
            except (KeyError, ValueError):
                out["problems"].append("a CSV row has no valid event_id")
        for e in self.events:
            r = by_id.get(e.event_id)
            if r is None:
                out["problems"].append(f"event {e.event_id} missing from CSV")
                continue
            if (int(r["frame"]), round(float(r["time_s"]), 3), r["visibility_label"]) != \
                    (e.frame, round(e.time_s, 3), e.visibility_label):
                out["problems"].append(f"event {e.event_id} differs between JSON and CSV")
        extra = set(by_id) - {e.event_id for e in self.events}
        if extra:
            out["problems"].append(f"CSV has ids not in JSON: {sorted(extra)}")
        out["ok"] = not out["problems"]
        return out

    def evidence_path(self, ev: Event) -> Path | None:
        """Absolute path of an event's evidence image if it exists on disk."""
        if not ev.evidence_frame:
            return None
        p = Path(ev.evidence_frame)
        p = p if p.is_absolute() else self.root / p
        return p if p.is_file() else None

    # ------------------------------------------------------------------ helpers
    def _unique_evidence_path(self, eid: int, frame: int) -> Path:
        """Never overwrite an existing image (it may belong to an event whose record was lost)."""
        base = self.evidence_dir / f"event_{eid:04d}_f{frame:06d}"
        p, n = base.with_suffix(".jpg"), 1
        while p.exists():
            n += 1
            p = Path(f"{base}_{n}.jpg")
        return p

    def _own_evidence_file(self, ev: Event) -> Path | None:
        p = self.evidence_path(ev)
        if p is None:
            return None
        try:
            p.resolve().relative_to(self.evidence_dir.resolve())
        except ValueError:
            return None                                       # only images this store created are ever moved
        return p

    @staticmethod
    def _remove_quietly(p: Path | None):
        try:
            if p is not None and p.exists():
                p.unlink()
        except OSError:
            pass
