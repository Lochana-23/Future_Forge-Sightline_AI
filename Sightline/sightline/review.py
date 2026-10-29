"""Human review of automated candidates (Stage D/E).

Automated candidate records are never modified. A reviewer's decisions live in a separate store
keyed by candidate_key, and the exported file shows both side by side with a clear human_ prefix.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from .events import FIELD_NAMES, VISIBILITY_LABELS, EventLog
from .evidence import parse_ids
from .fileio import atomic_write_csv, atomic_write_text, backup_file

STATUSES = ["unreviewed", "confirmed", "rejected", "uncertain"]
OBSTRUCTIONS = [
    "not assessed",
    "none visible",
    "stopped/parked vehicle",
    "moving vehicle",
    "pole, sign or structure",
    "other pedestrians",
    "other",
    "cannot tell",
]
DISCLAIMER = ("Human labels are reviewer judgements made from an elevated roadside camera image, not from a driver's "
              "viewpoint. They do not establish what any driver could or could not see, or driver attention.")


FLAG_HELP = {
    "short_event": "Very short (under 1 s): may be a single noisy detection.",
    "vehicle_near_stationary": "The vehicle barely moves: it may be parked or queued on the crosswalk.",
    "vehicle_class_flip": "The tracker changed the vehicle's class (e.g. car/bus/truck): class is unreliable.",
    "merged_multiple_vehicle_tracks": "Several vehicle track ids were merged: the tracker may have lost and re-found it.",
    "merged_multiple_pedestrian_tracks": "Several pedestrian ids were merged: one person may have been split, "
                                         "or two people merged.",
    "pedestrian_near_zone_not_inside": "The pedestrian is near the crosswalk, not yet on it (approach).",
    "pedestrian_far_from_vehicle": "The closest pedestrian is fairly far from the vehicle (heuristic threshold).",
    "overlaps_candidate_same_vehicle": "Another candidate in the same zone overlaps in time and uses the same "
                                       "vehicle track: this may be one long interaction split by pedestrian ids.",
    "overlaps_candidate_same_pedestrian": "Another candidate in the same zone overlaps in time and uses the same "
                                          "pedestrian track: this may be one long interaction split by vehicle ids.",
}


def explain_flags(flags: str) -> list[str]:
    """Plain-language explanations for the ';'-separated flag string."""
    out = []
    for f in [x for x in str(flags or "").split(";") if x]:
        m = re.fullmatch(r"(\d+)_likely_rider_boxes_excluded_in_zone", f)
        if m:
            out.append(f"{m.group(1)} person boxes overlapping detected motorcycles were treated as riders and "
                       f"excluded in this zone (riders of undetected motorcycles are NOT excluded).")
        else:
            out.append(FLAG_HELP.get(f, f))
    return out


@dataclass
class Review:
    candidate_key: str
    status: str = "unreviewed"
    visibility_label: str = "Unassessed"
    obstruction: str = "not assessed"
    notes: str = ""
    reviewer: str = ""
    reviewed_utc: str = ""
    # Describes the candidate at review time, so the review can be re-matched if candidates are regenerated.
    zone: str = ""
    frame_start: int = -1
    frame_end: int = -1
    vehicle_tracks: str = ""
    pedestrian_tracks: str = ""


REVIEW_FIELDS = [f.name for f in fields(Review)]
_DESCRIPTOR = {"zone", "frame_start", "frame_end", "vehicle_tracks", "pedestrian_tracks"}
EXPORT_FIELDS = [f for f in REVIEW_FIELDS if f != "candidate_key" and f not in _DESCRIPTOR]


def validate_review(r: Review) -> None:
    if not r.candidate_key:
        raise ValueError("candidate_key is required")
    if r.status not in STATUSES:
        raise ValueError(f"Invalid status: {r.status}")
    if r.visibility_label not in VISIBILITY_LABELS:
        raise ValueError(f"Invalid visibility label: {r.visibility_label}")
    if r.obstruction not in OBSTRUCTIONS:
        raise ValueError(f"Invalid obstruction: {r.obstruction}")


class ReviewStore:
    """Human decisions keyed by candidate_key. Reloads from disk before every save, writes atomically, and keeps
    a copy of the previous file in <folder>/backups/ so a bad save can be undone by hand."""

    def __init__(self, path):
        self.path = Path(path)
        self.reviews: dict[str, Review] = {}
        if self.path.is_file():
            self._load()

    def _load(self):
        """Raises ValueError (with the reason) for anything unreadable; the file is never modified."""
        self.reviews = {}
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.path} is not valid JSON ({e})") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("reviews", []), list):
            raise ValueError(f"{self.path} does not look like a reviews file (expected an object with 'reviews')")
        for i, row in enumerate(doc.get("reviews", []), start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{self.path}: review #{i} is not an object")
            known = {k: v for k, v in row.items() if k in REVIEW_FIELDS}
            try:
                r = Review(**known)
            except TypeError as e:
                raise ValueError(f"{self.path}: review #{i}: {e}") from None
            validate_review(r)
            self.reviews[r.candidate_key] = r

    def get(self, key: str) -> Review:
        return self.reviews.get(key, Review(candidate_key=key))

    def set(self, key: str, status: str, visibility_label: str = "Unassessed", obstruction: str = "not assessed",
            notes: str = "", reviewer: str = "", event=None) -> Review:
        """Record a decision in memory. Pass the candidate `event` so its identity is stored with the review."""
        if status in ("rejected", "unreviewed"):
            visibility_label, obstruction = "Unassessed", "not assessed"   # a label on a non-event is meaningless
        r = Review(key, status, visibility_label, obstruction, notes, reviewer,
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if event is not None:
            r.zone, r.frame_start, r.frame_end = event.zone, int(event.frame_start), int(event.frame_end)
            r.vehicle_tracks, r.pedestrian_tracks = str(event.vehicle_tracks), str(event.pedestrian_tracks)
        validate_review(r)
        if status == "unreviewed":
            self.reviews.pop(key, None)
        else:
            self.reviews[key] = r
        return r

    def decide(self, key: str, *args, **kw) -> Review:
        """Reload from disk, apply one decision, save. Use this from the UI so two open sessions cannot
        overwrite each other's earlier decisions with stale data."""
        if self.path.is_file():
            self._load()
        r = self.set(key, *args, **kw)
        self.save()
        return r

    def save(self) -> Path:
        """Atomic write; the previous version is kept as backups/reviews_previous.json."""
        doc = {"disclaimer": DISCLAIMER, "reviews": [asdict(r) for r in self.reviews.values()]}
        if self.path.is_file():
            bdir = self.path.parent / "backups"
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, bdir / "reviews_previous.json")
        return atomic_write_text(self.path, json.dumps(doc, indent=2))

    def orphans(self, events) -> list[str]:
        """Review keys that match no current candidate (e.g. candidates were regenerated with new settings)."""
        live = {e.candidate_key for e in events}
        return sorted(k for k in self.reviews if k not in live)

    # ---- regeneration ---------------------------------------------------------
    def backfill(self, events) -> int:
        """Add the candidate description to reviews that were saved before descriptions existed. Memory only."""
        by_key = {e.candidate_key: e for e in events}
        n = 0
        for k, r in self.reviews.items():
            e = by_key.get(k)
            if e is not None and r.frame_start < 0 and e.frame_start >= 0:
                r.zone, r.frame_start, r.frame_end = e.zone, int(e.frame_start), int(e.frame_end)
                r.vehicle_tracks, r.pedestrian_tracks = str(e.vehicle_tracks), str(e.pedestrian_tracks)
                n += 1
        return n

    def reconcile(self, events) -> dict:
        """Compare saved reviews with the current candidates.

        A review whose key is no longer present is matched to a new candidate ONLY if it has a saved description and
        exactly one unclaimed candidate has the same zone, an overlapping frame span, and at least one shared
        vehicle track id AND one shared pedestrian track id. Matches are proposals: nothing is changed until
        apply_rekeys(). Anything else is reported as ambiguous or unmatched, never attached silently."""
        live = {e.candidate_key: e for e in events}
        claimed = {k for k in self.reviews if k in live}
        proposals, ambiguous, unmatched = [], [], []
        for k in sorted(set(self.reviews) - set(live)):
            r = self.reviews[k]
            if r.frame_start < 0 or not r.zone:
                unmatched.append({"key": k, "reason": "saved before candidate descriptions existed; cannot re-match"})
                continue
            rv, rp = set(parse_ids(r.vehicle_tracks)), set(parse_ids(r.pedestrian_tracks))
            hits = []
            for e in events:
                if e.candidate_key in claimed or e.zone != r.zone or e.frame_start < 0:
                    continue
                if e.frame_end < r.frame_start or e.frame_start > r.frame_end:
                    continue
                if rv & set(parse_ids(e.vehicle_tracks)) and rp & set(parse_ids(e.pedestrian_tracks)):
                    hits.append(e)
            if len(hits) == 1:
                proposals.append({"old_key": k, "new_key": hits[0].candidate_key, "time_s": hits[0].time_s,
                                  "status": r.status, "reason": "same zone, overlapping frames, shared vehicle and "
                                  "pedestrian track ids"})
                claimed.add(hits[0].candidate_key)
            elif hits:
                ambiguous.append({"key": k, "candidates": [h.candidate_key for h in hits]})
            else:
                unmatched.append({"key": k, "reason": "no current candidate shares its zone, time span and track ids"})
        # two reviews proposing the same new candidate are ambiguous, not proposals
        seen = {}
        for p in proposals:
            seen.setdefault(p["new_key"], []).append(p)
        final = []
        for new, ps in seen.items():
            if len(ps) == 1:
                final.append(ps[0])
            else:
                for p in ps:
                    ambiguous.append({"key": p["old_key"], "candidates": [new]})
        return {"exact": len([k for k in self.reviews if k in live]), "proposals": final,
                "ambiguous": ambiguous, "unmatched": unmatched}

    def apply_rekeys(self, proposals, events) -> Path | None:
        """Move reviews to their proposed new candidate keys (after a timestamped backup). Returns the backup."""
        bk = backup_file(self.path, self.path.parent / "backups", "prerekey") if self.path.is_file() else None
        by_key = {e.candidate_key: e for e in events}
        for p in proposals:
            r = self.reviews.pop(p["old_key"], None)
            e = by_key.get(p["new_key"])
            if r is None or e is None:
                continue
            r.candidate_key = p["new_key"]
            r.zone, r.frame_start, r.frame_end = e.zone, int(e.frame_start), int(e.frame_end)
            r.vehicle_tracks, r.pedestrian_tracks = str(e.vehicle_tracks), str(e.pedestrian_tracks)
            self.reviews[r.candidate_key] = r
        self.save()
        return bk


def load_candidates(path) -> list:
    """Candidate events from candidates.json (written by make_candidates)."""
    return EventLog.from_json(path).events


def summary(events, store: ReviewStore) -> dict:
    n = len(events)
    st = [store.get(e.candidate_key).status for e in events]
    labels = [store.get(e.candidate_key).visibility_label for e in events
              if store.get(e.candidate_key).status == "confirmed"]
    decided = st.count("confirmed") + st.count("rejected")
    return {
        "candidates": n,
        "reviewed": n - st.count("unreviewed"),
        "confirmed": st.count("confirmed"),
        "rejected": st.count("rejected"),
        "uncertain": st.count("uncertain"),
        "unreviewed": st.count("unreviewed"),
        "confirmed_share_of_decided": round(st.count("confirmed") / decided, 3) if decided else None,
        "visibility_labels_among_confirmed": {lab: labels.count(lab) for lab in VISIBILITY_LABELS if labels.count(lab)},
        "orphan_reviews": len(store.orphans(events)),
        "note": "confirmed_share_of_decided is a reviewer-based estimate of how many candidates were real "
                "vehicle-pedestrian moments; it is NOT detector accuracy and covers only the reviewed candidates.",
    }


def export_reviewed(events, store: ReviewStore, out_dir, video_name: str = "") -> tuple[Path, Path]:
    """Write reviewed_events.csv/json: automated columns, then human_* columns. Automated data is unchanged."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    human_cols = ["human_" + f for f in EXPORT_FIELDS]
    rows = []
    for e in events:
        row = {k: getattr(e, k) for k in FIELD_NAMES}
        r = store.get(e.candidate_key)
        for f in EXPORT_FIELDS:
            row["human_" + f] = getattr(r, f)
        rows.append(row)
    csv_p = out_dir / "reviewed_events.csv"
    atomic_write_csv(csv_p, FIELD_NAMES + human_cols, rows)
    json_p = out_dir / "reviewed_events.json"
    atomic_write_text(json_p, json.dumps({"video": video_name, "disclaimer": DISCLAIMER,
                                          "summary": summary(events, store), "events": rows}, indent=2))
    return csv_p, json_p
