"""Rule-based candidate events from tracks + crosswalk zones (Stage C3). Pure logic.

An event is a (vehicle, pedestrian) PAIR in one zone:
  - the vehicle overlaps the zone (>= veh_overlap of its box area),
  - the pedestrian's foot point is inside the zone or within ped_margin_px of it
    (so people APPROACHING a crosswalk count; such events are flagged),
  - and the pedestrian's foot point is within max_gap_ped_heights pedestrian-heights of the vehicle box
    (a parked vehicle with people walking past it gives one event per person, not one giant event).
Everything is in image pixels, not metres. Defaults are heuristics tuned on ONE hand-labelled moment;
they are NOT validated. All results are automated CANDIDATES for human review, never ground truth.
It is not a visibility judgement and not a conflict/violation judgement.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .tracks import TrackRow

VEHICLES = {"car", "bus", "truck", "motorcycle"}
TWO_WHEELERS = {"motorcycle", "bicycle"}


# ---- geometry ---------------------------------------------------------------
def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b):
    i = inter_area(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def point_in_polygon(pt, poly) -> bool:
    """Ray casting. Points exactly on an edge may go either way."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def dist_point_to_box(pt, b) -> float:
    dx = max(b[0] - pt[0], 0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0, pt[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def dist_point_to_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5


def dist_point_to_polygon(pt, poly) -> float:
    """0 if inside, else distance (pixels) to the nearest edge."""
    if point_in_polygon(pt, poly):
        return 0.0
    return min(dist_point_to_segment(pt, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def box_fraction_in_polygon(box, poly, n: int = 8) -> float:
    """Approximate fraction of the box area inside the polygon (n x n sample grid)."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if x2 < min(xs) or x1 > max(xs) or y2 < min(ys) or y1 > max(ys):
        return 0.0                                   # quick reject: no bounding-box overlap
    hit = 0
    for i in range(n):
        for j in range(n):
            if point_in_polygon((x1 + (i + 0.5) * (x2 - x1) / n, y1 + (j + 0.5) * (y2 - y1) / n), poly):
                hit += 1
    return hit / (n * n)


def box_of(r: TrackRow):
    return (r.x1, r.y1, r.x2, r.y2)


def foot(r: TrackRow):
    return ((r.x1 + r.x2) / 2, r.y2)


def centre(r: TrackRow):
    return ((r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2)


# ---- zones ------------------------------------------------------------------
def load_zones(path) -> dict[str, list]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = {}
    for z in doc["zones"]:
        poly = [tuple(p) for p in z["polygon"]]
        if len(poly) < 3:
            raise ValueError(f"Zone '{z['name']}' has fewer than 3 points")
        zones[z["name"]] = poly
    if not zones:
        raise ValueError("No zones defined")
    return zones


# ---- rider suspects ----------------------------------------------------------
def rider_suspects(rows: list[TrackRow], thr: float = 0.3) -> set[tuple[int, int]]:
    """(frame, track_id) of person boxes that overlap a motorcycle/bicycle box in the same frame.

    Only catches riders whose two-wheeler was ALSO detected; riders of missed two-wheelers remain.
    """
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    out = set()
    for f, rs in by_frame.items():
        tw = [box_of(r) for r in rs if r.cls in TWO_WHEELERS]
        if not tw:
            continue
        for r in rs:
            if r.cls == "person":
                pb = box_of(r)
                a = box_area(pb)
                if a > 0 and any(inter_area(pb, t) / a >= thr for t in tw):
                    out.add((f, r.track_id))
    return out


# ---- candidates ---------------------------------------------------------------
@dataclass
class PairObs:
    frame: int
    veh: TrackRow
    ped: TrackRow
    ped_inside: bool          # False if the pedestrian was only NEAR the zone
    gap_px: float             # pedestrian foot point to vehicle box, image pixels
    gap_h: float              # same gap in units of the pedestrian's box height


@dataclass
class Candidate:
    zone: str
    start_frame: int
    end_frame: int
    vehicle_class: str
    vehicle_tracks: list
    pedestrian_tracks: list
    n_obs: int
    evidence_frame: int
    min_px_gap: float
    det_conf_mean: float
    flags: list = field(default_factory=list)
    riders_excluded: int = 0
    gap_ped_heights: float = 0.0
    vehicle_box: tuple = ()
    ped_box: tuple = ()
    obs: list = field(default_factory=list, repr=False)

    def key(self) -> str:
        """Stable-ish id so human reviews survive re-runs that reproduce the same event."""
        raw = f"{self.zone}|{min(self.pedestrian_tracks)}|{min(self.vehicle_tracks)}|{self.start_frame}"
        return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _zone_members(rs, f, poly, suspects, veh_overlap, ped_margin_px):
    """(peds as [(row, inside)], vehicles, number of rider boxes excluded) for one zone in one frame."""
    peds, excluded = [], 0
    for r in rs:
        if r.cls == "person" and r.track_id >= 0:      # an id-less detection cannot be told apart from others
            d = dist_point_to_polygon(foot(r), poly)
            if d <= ped_margin_px:
                if (f, r.track_id) in suspects:
                    excluded += 1
                else:
                    peds.append((r, d == 0.0))
    vehs = [r for r in rs if r.cls in VEHICLES and r.track_id >= 0
            and box_fraction_in_polygon(box_of(r), poly) >= veh_overlap]
    return peds, vehs, excluded


def _build(zone, obs: list[PairObs], fps, far_ped_heights) -> Candidate:
    obs = sorted(obs, key=lambda o: (o.frame, o.gap_px))
    best = min(obs, key=lambda o: o.gap_px)
    veh_seen = {(o.frame, o.veh.track_id): o.veh for o in obs}
    classes = Counter(v.cls for v in veh_seen.values())
    confs = [o.veh.conf for o in obs] + [o.ped.conf for o in obs]
    cs = [centre(v) for v in veh_seen.values()]
    disp = max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a in cs for b in cs)
    veh_ids = sorted({o.veh.track_id for o in obs})
    ped_ids = sorted({o.ped.track_id for o in obs})
    flags = []
    if (obs[-1].frame - obs[0].frame) / fps < 1.0:
        flags.append("short_event")
    if disp < 20:
        flags.append("vehicle_near_stationary")
    if len(classes) > 1:
        flags.append("vehicle_class_flip")
    if len(veh_ids) > 1:
        flags.append("merged_multiple_vehicle_tracks")
    if len(ped_ids) > 1:
        flags.append("merged_multiple_pedestrian_tracks")
    if not any(o.ped_inside for o in obs):
        flags.append("pedestrian_near_zone_not_inside")
    if far_ped_heights and best.gap_h > far_ped_heights:
        flags.append("pedestrian_far_from_vehicle")      # heuristic, threshold not validated
    return Candidate(zone, obs[0].frame, obs[-1].frame, classes.most_common(1)[0][0], veh_ids, ped_ids,
                     len({(o.frame, o.veh.track_id, o.ped.track_id) for o in obs}), best.frame,
                     round(best.gap_px, 1), round(sum(confs) / len(confs), 3), flags, 0, round(best.gap_h, 2),
                     box_of(best.veh), box_of(best.ped), obs)


def _same_vehicle(a: PairObs, b: PairObs, handover_iou: float) -> bool:
    return a.veh.track_id == b.veh.track_id or iou(box_of(a.veh), box_of(b.veh)) >= handover_iou


def _same_ped(a: PairObs, b: PairObs) -> bool:
    if a.ped.track_id == b.ped.track_id:
        return True
    (ax, ay), (bx, by) = foot(a.ped), foot(b.ped)
    h = max(a.ped.y2 - a.ped.y1, b.ped.y2 - b.ped.y1, 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= 0.6 * h


def find_candidates(rows: list[TrackRow], zones: dict, fps: float, gap_s: float = 1.0, min_obs: int = 2,
                    exclude_riders: bool = True, handover_iou: float = 0.3, far_ped_heights: float = 2.0,
                    veh_overlap: float = 0.15, ped_margin_px: float = 120.0,
                    max_gap_ped_heights: float = 3.0) -> list[Candidate]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    suspects = rider_suspects(rows) if exclude_riders else set()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    gap_frames = max(1, int(round(gap_s * fps)))

    pair_obs = defaultdict(list)         # (zone, vehicle track, pedestrian track) -> [PairObs]
    riders_by_zone = Counter()
    for f in sorted(by_frame):
        rs = by_frame[f]
        for zname, poly in zones.items():
            peds_m, vehs, ex = _zone_members(rs, f, poly, suspects, veh_overlap, ped_margin_px)
            riders_by_zone[zname] += ex
            for v in vehs:
                vb = box_of(v)
                for p, inside in peds_m:
                    d = dist_point_to_box(foot(p), vb)
                    gh = d / max(p.y2 - p.y1, 1.0)
                    if gh <= max_gap_ped_heights:
                        pair_obs[(zname, v.track_id, p.track_id)].append(PairObs(f, v, p, inside, d, gh))

    runs = []
    for (zname, _, _), obs in pair_obs.items():
        obs.sort(key=lambda o: o.frame)
        run = [obs[0]]
        for o in obs[1:]:
            if o.frame - run[-1].frame > gap_frames:
                runs.append((zname, run))
                run = [o]
            else:
                run.append(o)
        runs.append((zname, run))
    runs = [(z, r) for z, r in runs if len(r) >= min_obs]

    # merge fragments caused by tracker identity switches (same zone, same vehicle, same pedestrian)
    runs.sort(key=lambda zr: (zr[0], zr[1][0].frame))
    merged = []
    for z, run in runs:
        target = None
        for m in merged:
            if m[0] != z:
                continue
            nearest = min(m[1], key=lambda o: abs(o.frame - run[0].frame))
            if abs(nearest.frame - run[0].frame) <= gap_frames and _same_vehicle(nearest, run[0], handover_iou) \
                    and _same_ped(nearest, run[0]):
                target = m
                break
        if target:
            target[1].extend(run)
        else:
            merged.append((z, list(run)))

    out = []
    for z, run in merged:
        c = _build(z, run, fps, far_ped_heights)
        c.riders_excluded = riders_by_zone.get(z, 0)
        out.append(c)
    out.sort(key=lambda c: (c.start_frame, c.zone))
    annotate_overlaps(out)
    return out


def annotate_overlaps(cands: list) -> None:
    """Flag candidates that overlap another in the same zone AND share a vehicle or a pedestrian track id.

    Such pairs are often ONE prolonged interaction that the pairing rule split (several pedestrians crossing in
    front of one vehicle, or one pedestrian passing several vehicles). They are not merged automatically, because
    that can also be two genuinely separate moments; the flag tells the reviewer to look at them together.
    """
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if a.zone != b.zone or b.start_frame > a.end_frame or a.start_frame > b.end_frame:
                continue
            if set(a.vehicle_tracks) & set(b.vehicle_tracks):
                for c in (a, b):
                    if "overlaps_candidate_same_vehicle" not in c.flags:
                        c.flags.append("overlaps_candidate_same_vehicle")
            if set(a.pedestrian_tracks) & set(b.pedestrian_tracks):
                for c in (a, b):
                    if "overlaps_candidate_same_pedestrian" not in c.flags:
                        c.flags.append("overlaps_candidate_same_pedestrian")


def unique_keys(cands: list) -> list[str]:
    """candidate_key for each candidate, guaranteed unique within one run (a suffix is added on a collision)."""
    seen, out = {}, []
    for c in cands:
        k = c.key()
        seen[k] = seen.get(k, 0) + 1
        out.append(k if seen[k] == 1 else f"{k}-{seen[k]}")
    return out


# ---- diagnostics ----------------------------------------------------------------
def diagnose(rows: list[TrackRow], zones: dict, exclude_riders: bool = True, veh_overlap: float = 0.15,
             ped_margin_px: float = 120.0) -> dict:
    """Per zone: how many processed frames had a pedestrian (inside/near), a vehicle, and both.

    Explains low recall: zero 'both' frames means the rule never had a chance to fire.
    """
    suspects = rider_suspects(rows) if exclude_riders else set()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    out = {z: {"frames_with_pedestrian_inside": 0, "frames_with_pedestrian_inside_or_near": 0,
               "frames_with_vehicle": 0, "frames_with_both": 0, "rider_boxes_excluded": 0,
               "id_less_person_boxes_skipped": 0} for z in zones}
    for f, rs in by_frame.items():
        for z, poly in zones.items():
            peds, vehs, ex = _zone_members(rs, f, poly, suspects, veh_overlap, ped_margin_px)
            out[z]["rider_boxes_excluded"] += ex
            out[z]["id_less_person_boxes_skipped"] += sum(
                1 for r in rs if r.cls == "person" and r.track_id < 0
                and dist_point_to_polygon(foot(r), poly) <= ped_margin_px)
            out[z]["frames_with_pedestrian_inside"] += any(i for _, i in peds)
            out[z]["frames_with_pedestrian_inside_or_near"] += bool(peds)
            out[z]["frames_with_vehicle"] += bool(vehs)
            out[z]["frames_with_both"] += bool(peds and vehs)
    out["_processed_frames"] = len(by_frame)
    return out


def explain_frame(rows: list[TrackRow], zones: dict, frame: int, window: int = 6, exclude_riders: bool = True,
                  veh_overlap: float = 0.15, ped_margin_px: float = 120.0,
                  max_gap_ped_heights: float = 3.0) -> list[str]:
    """Human-readable list of what the tracker saw near `frame` and how each object relates to each zone."""
    suspects = rider_suspects(rows) if exclude_riders else set()
    frames = sorted({r.frame for r in rows if abs(r.frame - frame) <= window})
    if not frames:
        return [f"No processed frames within {window} frames of {frame} (tracking ran with a stride / window "
                f"that skips it)."]
    lines = [f"(vehicle counts if >= {veh_overlap:.0%} of its box overlaps a zone; pedestrian counts if its foot "
             f"point is inside or within {ped_margin_px:.0f}px; a pair needs a gap <= {max_gap_ped_heights} "
             f"pedestrian heights)"]
    for f in frames:
        lines.append(f"frame {f}:")
        frame_rows = [r for r in rows if r.frame == f]
        for r in sorted(frame_rows, key=lambda r: (r.cls, r.track_id)):
            rel = []
            if r.cls == "person":
                for z, poly in zones.items():
                    d = dist_point_to_polygon(foot(r), poly)
                    if d == 0.0:
                        rel.append(f"{z}(inside)")
                    elif d <= ped_margin_px:
                        rel.append(f"{z}(near {d:.0f}px)")
                pos = f"foot=({foot(r)[0]:.0f},{foot(r)[1]:.0f})"
            elif r.cls in VEHICLES:
                for z, poly in zones.items():
                    fr = box_fraction_in_polygon(box_of(r), poly)
                    if fr >= veh_overlap:
                        rel.append(f"{z}({fr:.0%} overlap)")
                pos = f"centre=({centre(r)[0]:.0f},{centre(r)[1]:.0f})"
            else:
                continue
            note = " [likely rider: excluded]" if (f, r.track_id) in suspects else ""
            lines.append(f"  {r.cls}#{r.track_id} conf={r.conf:.2f} {pos} zones: {', '.join(rel) or 'none'}{note}")
        for z, poly in zones.items():
            peds, vehs, _ = _zone_members(frame_rows, f, poly, suspects, veh_overlap, ped_margin_px)
            for v in vehs:
                for p, inside in peds:
                    d = dist_point_to_box(foot(p), box_of(v))
                    gh = d / max(p.y2 - p.y1, 1.0)
                    tag = "PAIR" if gh <= max_gap_ped_heights else "too far"
                    lines.append(f"  {tag} in {z}: {v.cls}#{v.track_id} - person#{p.track_id}: gap {d:.0f}px "
                                 f"= {gh:.1f} ped heights ({'inside' if inside else 'near'})")
    return lines
