"""Manual event log with CSV/JSON export (Stage B).

Every event here is created by a HUMAN (source="manual"). Automated detections will be
added in Stage C with source="auto" and kept separate from human-confirmed labels.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from .fileio import atomic_write_csv, atomic_write_text

# Reviewer-defined labels. They describe what the reviewer judges from the stated perspective.
VISIBILITY_LABELS = ["Unassessed", "Clear", "Partial", "Blocked", "Uncertain"]
PERSPECTIVE_DEFAULT = "Roadside elevated camera view (NOT the driver's viewpoint)"
LIMITATIONS_DEFAULT = (
    "Judged from camera image only. Image overlap does not prove a driver could not see "
    "the pedestrian; geometry and driver attention are not assessed."
)


@dataclass
class Event:
    event_id: int
    frame: int
    time_s: float
    source: str = "manual"            # "manual" now; "auto" from Stage C
    review_status: str = "manual"     # manual / confirmed / rejected / uncertain (Stage D+)
    event_type: str = "pedestrian-vehicle interaction"
    visibility_label: str = "Unassessed"
    perspective: str = PERSPECTIVE_DEFAULT
    model_confidence: str = ""        # empty for manual events
    notes: str = ""
    limitations: str = LIMITATIONS_DEFAULT
    evidence_frame: str = ""
    frame_start: int = -1             # first frame of an automated candidate; -1 for manual events
    frame_end: int = -1               # last frame of an automated candidate; -1 for manual events
    zone: str = ""
    vehicle_class: str = ""
    vehicle_tracks: str = ""
    pedestrian_tracks: str = ""
    flags: str = ""                   # machine-generated caution flags, ';'-separated
    candidate_key: str = ""           # stable id linking an automated candidate to its human review
    vehicle_box: str = ""             # "x1,y1,x2,y2" image pixels at the evidence frame
    pedestrian_box: str = ""
    evidence_sheet: str = ""          # path of a multi-frame contact sheet
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    video: str = ""                   # file name of the video the event belongs to (added in schema 2)


FIELD_NAMES = [f.name for f in fields(Event)]
_INT_FIELDS = {"event_id", "frame", "frame_start", "frame_end"}
_REQUIRED = ("event_id", "frame", "time_s")


def event_from_row(row, where: str = "event") -> Event:
    """Build an Event from a dict (JSON row) with validation. Raises ValueError saying exactly what is wrong.
    Unknown keys are ignored (forward compatibility); missing optional keys take their defaults."""
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected an object, got {type(row).__name__}")
    missing = [k for k in _REQUIRED if row.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{where}: missing required field(s) {missing}")
    known = {k: v for k, v in row.items() if k in FIELD_NAMES}
    try:
        for k in _INT_FIELDS:
            if k in known and known[k] not in (None, ""):
                if isinstance(known[k], bool) or (isinstance(known[k], float) and not known[k].is_integer()):
                    raise ValueError(k)
                known[k] = int(known[k])
        known["time_s"] = float(known["time_s"])
    except (TypeError, ValueError):
        raise ValueError(f"{where}: a numeric field is not a number (event_id/frame/time_s in {row!r:.120})") from None
    if not math.isfinite(known["time_s"]) or known["frame"] < 0 or known["event_id"] < 1:
        raise ValueError(f"{where}: event_id must be >= 1, frame >= 0 and time_s finite")
    for k, v in list(known.items()):
        if k not in _INT_FIELDS and k != "time_s" and v is None:
            known[k] = ""
    if known.get("visibility_label", "Unassessed") not in VISIBILITY_LABELS:
        raise ValueError(f"{where}: unknown visibility_label {known['visibility_label']!r}")
    return Event(**known)


class EventLog:
    """An in-memory list of events with validated add/delete and atomic CSV/JSON writers.

    Used for automated candidates (make_candidates). Manual, human-entered events use `store.EventStore`, which
    adds persistence, permanent ids and backups on top of the same Event records.
    """

    def __init__(self):
        self.events: list[Event] = []

    def add(self, frame: int, fps: float, **kw) -> Event:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if frame < 0:
            raise ValueError("frame must be >= 0")
        label = kw.get("visibility_label", "Unassessed")
        if label not in VISIBILITY_LABELS:
            raise ValueError(f"Invalid visibility label: {label}")
        next_id = max((e.event_id for e in self.events), default=0) + 1
        ev = Event(event_id=next_id, frame=int(frame), time_s=round(frame / fps, 3), **kw)
        self.events.append(ev)
        return ev

    def delete(self, event_id: int) -> bool:
        before = len(self.events)
        self.events = [e for e in self.events if e.event_id != event_id]
        return len(self.events) < before

    def rows(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def to_csv(self, path) -> Path:
        return atomic_write_csv(path, FIELD_NAMES, self.rows())

    def to_json(self, path, video_name: str = "") -> Path:
        doc = {
            "video": video_name,
            "note": "Reviewer/candidate events. Labels are reviewer-defined, not ground truth.",
            "events": self.rows(),
        }
        return atomic_write_text(path, json.dumps(doc, indent=2))

    @classmethod
    def from_json(cls, path) -> "EventLog":
        """Raises OSError (cannot read) or ValueError (not valid / wrong shape). Never returns a silently partial log."""
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON ({e})") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("events", []), list):
            raise ValueError(f"{path} does not look like an events file (expected an object with an 'events' list)")
        log = cls()
        seen = set()
        for i, row in enumerate(doc.get("events", []), start=1):
            ev = event_from_row(row, f"{path} event #{i}")
            if ev.event_id in seen:
                raise ValueError(f"{path}: duplicate event_id {ev.event_id}")
            seen.add(ev.event_id)
            log.events.append(ev)
        return log
