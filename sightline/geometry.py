"""EXPERIMENTAL line-of-sight geometry and what-if (Stages F and G). Numpy only.

Honesty rules built into this module:
  * Nothing is computed unless a calibration passes a gate (>= 5 real ground points, error estimate, not an
    example file, small leave-one-out reprojection error). Otherwise the status is NOT_ASSESSABLE.
  * Inputs are separated in every report: MEASURED (calibration points the user measured elsewhere),
    USER-ENTERED ASSUMPTIONS (eye position/height, object heights, obstacle footprints),
    MODEL ESTIMATES (the Monte Carlo result), HYPOTHETICAL (what-if).
  * The ground-plane homography only maps points ON THE ROAD SURFACE. Elevated points are not mapped by it;
    heights are separate user assumptions.
  * The result is geometric visibility from an ASSUMED eye point. It says nothing about whether the driver looked,
    reacted or would have behaved differently, and a what-if does not predict behaviour or crash outcomes.
  * The label is reported only if >= 80% of sensitivity samples agree; otherwise it is "Uncertain".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GATE_MIN_POINTS = 5
GATE_MAX_LOO_MEAN_M = 0.75
GATE_MAX_LOO_MAX_M = 1.5
ROBUSTNESS = 0.8
BODY_FRACTIONS = (0.15, 0.35, 0.55, 0.75, 0.9, 1.0)   # heights on the pedestrian, as a share of body height

DISCLAIMER = ("EXPERIMENTAL. Geometric line-of-sight from an ASSUMED eye point on a flat ground plane. It does not "
              "show what a driver saw, noticed or would have done. A what-if moves one object in the model only; it "
              "does not predict behaviour or any change in crashes.")


# ---- homography ------------------------------------------------------------------
def _norm_T(pts):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def fit_homography(img_pts, world_pts) -> np.ndarray:
    """Normalised DLT: maps image pixels (x,y) on the ground plane to world metres (X,Y)."""
    a = np.asarray(img_pts, float)
    b = np.asarray(world_pts, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 4:
        raise ValueError("need at least 4 matching (x,y) pairs")
    Ta, Tb = _norm_T(a), _norm_T(b)
    ah = (Ta @ np.c_[a, np.ones(len(a))].T).T
    bh = (Tb @ np.c_[b, np.ones(len(b))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(ah, bh):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(Tb) @ vt[-1].reshape(3, 3) @ Ta
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        raise ValueError("degenerate homography (points are not a usable ground-plane calibration)")
    return H / H[2, 2]


def apply_h(H, pt) -> tuple[float, float]:
    """Image pixel -> world metres. Refuses points at or beyond the horizon (w <= 0) instead of returning nonsense."""
    if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
        raise ValueError("pixel coordinates must be finite numbers")
    v = H @ np.array([pt[0], pt[1], 1.0])
    if v[2] <= 1e-9:
        raise ValueError(f"pixel ({pt[0]:.0f}, {pt[1]:.0f}) is at or beyond the horizon of the ground plane "
                         "(outside the calibrated area): choose a point on the road surface")
    return float(v[0] / v[2]), float(v[1] / v[2])


def loo_errors(img_pts, world_pts) -> list[float]:
    """Leave-one-out reprojection error (metres) for each calibration point."""
    a, b = np.asarray(img_pts, float), np.asarray(world_pts, float)
    errs = []
    for i in range(len(a)):
        keep = [j for j in range(len(a)) if j != i]
        H = fit_homography(a[keep], b[keep])
        try:
            p = apply_h(H, a[i])
        except ValueError:
            errs.append(float("inf"))      # a held-out point cannot be mapped at all: the calibration is unusable
            continue
        errs.append(math.hypot(p[0] - b[i][0], p[1] - b[i][1]))
    return errs


# ---- calibration gate ---------------------------------------------------------------
def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_calibration(doc: dict) -> dict:
    """Return {'status','reasons','loo_error_m','n_points','sigma_cal_m'} and (when OK) an 'H' key."""
    reasons = []
    if not isinstance(doc, dict):
        return {"status": "NOT_ASSESSABLE", "reasons": ["calibration file is not a JSON object"], "n_points": 0}
    if doc.get("example_only"):
        reasons.append("this is the example file (example_only is true): replace it with real measurements")
    if not str(doc.get("source", "")).strip():
        reasons.append("'source' is empty: say how the world coordinates were measured")
    if doc.get("units") != "metres":
        reasons.append("'units' must be exactly 'metres'")
    me = doc.get("measurement_error_m")
    if isinstance(me, bool) or not isinstance(me, (int, float)) or not (0 < me <= 1.0):
        reasons.append("'measurement_error_m' must be a number in (0, 1.0]: your honest error estimate per coordinate")
    pts = doc.get("ground_points") or []
    img, wor = [], []
    for i, p in enumerate(pts):
        try:
            im, wo = p["image"], p["world"]
            assert len(im) == 2 and len(wo) == 2
            vals = [float(im[0]), float(im[1]), float(wo[0]), float(wo[1])]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("not finite")
            img.append(vals[:2])
            wor.append(vals[2:])
        except Exception:
            reasons.append(f"ground point #{i + 1} is malformed (need finite numbers: 'image': [x,y] and "
                           "'world': [X,Y])")
    size = doc.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in size):
        for i, (x, y) in enumerate(img):
            if not (0 <= x <= size[0] and 0 <= y <= size[1]):
                reasons.append(f"ground point #{i + 1} image position ({x:g}, {y:g}) is outside image_size {list(size)}")
    n = len(img)
    if n < GATE_MIN_POINTS:
        reasons.append(f"need at least {GATE_MIN_POINTS} ground points, found {n}")
    out = {"n_points": n, "reasons": reasons}
    if n >= GATE_MIN_POINTS and not reasons:
        a, b = np.array(img), np.array(wor)
        if len({tuple(r) for r in a.round(3)}) < n:
            reasons.append("duplicate image points")
        else:
            for name, arr in (("image", a), ("world", b)):
                ev = np.linalg.eigvalsh(np.cov(arr.T))
                if ev[-1] <= 0 or ev[0] / ev[-1] < 0.01:
                    reasons.append(f"{name} points are (nearly) in a line: spread them over the area")
        if not reasons:
            try:
                errs = loo_errors(a, b)
            except (np.linalg.LinAlgError, ValueError) as e:
                errs = None
                reasons.append(f"calibration could not be fitted ({e})")
            if errs is not None and not all(math.isfinite(e) for e in errs):
                reasons.append("leave-one-out error too large (a held-out point could not be mapped, so the points do not "
                               "form a consistent ground plane): re-measure or add points")
            elif errs is not None:
                mean_e, max_e = float(np.mean(errs)), float(np.max(errs))
                out["loo_error_m"] = {"mean": round(mean_e, 3), "max": round(max_e, 3),
                                      "per_point": [round(e, 3) for e in errs]}
                if mean_e > GATE_MAX_LOO_MEAN_M or max_e > GATE_MAX_LOO_MAX_M:
                    reasons.append(f"leave-one-out error too large (mean {mean_e:.2f} m, max {max_e:.2f} m; limits "
                                   f"{GATE_MAX_LOO_MEAN_M}/{GATE_MAX_LOO_MAX_M} m): re-measure or add points")
                else:
                    H = fit_homography(a, b)
                    try:
                        for p in a:
                            apply_h(H, p)                      # every calibration point must be in front of the horizon
                    except ValueError as e:
                        reasons.append(f"calibration is inconsistent: {e}")
                    else:
                        out["sigma_cal_m"] = round(max(mean_e, float(me)), 3)
                        out["H"] = H
    out["status"] = "EXPERIMENTAL_OK" if not out["reasons"] and "H" in out else "NOT_ASSESSABLE"
    return out


# ---- line of sight ------------------------------------------------------------------
def _pip_vec(xs, ys, poly):
    inside = np.zeros(xs.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= cond & (xs < xint)
    return inside


def segment_blocked(eye, target, polygon, height) -> bool:
    """True if the 3-D segment eye->target passes through the prism (polygon footprint, 0..height)."""
    (xe, ye, ze), (xt, yt, zt) = eye, target
    length = math.sqrt((xt - xe) ** 2 + (yt - ye) ** 2 + (zt - ze) ** 2)
    n = int(max(200, math.ceil(length / 0.05)))
    t = np.linspace(0.0, 1.0, n)
    xs, ys, zs = xe + (xt - xe) * t, ye + (yt - ye) * t, ze + (zt - ze) * t
    m = (zs >= 0) & (zs <= height)
    return bool(m.any() and _pip_vec(xs[m], ys[m], polygon).any())


def _range(v, name):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v]
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(q, (int, float)) and math.isfinite(q)
                                                                  for q in v) and 0 < v[0] <= v[1]):
        raise ValueError(f"'{name}' must be a positive number or [low, high] with low <= high")
    return float(v[0]), float(v[1])


def validate_scenario(doc: dict) -> None:
    if not isinstance(doc, dict):
        raise ValueError("scenario must be a JSON object")
    for k in ("eye_ground_px", "pedestrian_foot_px"):
        v = doc.get(k)
        if not (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) for q in v)):
            raise ValueError(f"'{k}' must be [x, y] finite numbers in image pixels")
    n = doc.get("samples", 300)
    if isinstance(n, bool) or not isinstance(n, int) or not (20 <= n <= 5000):
        raise ValueError("'samples' must be a whole number between 20 and 5000")
    _range(doc.get("eye_height_m"), "eye_height_m")
    _range(doc.get("pedestrian_height_m"), "pedestrian_height_m")
    obs = doc.get("obstacles")
    if not isinstance(obs, list):
        raise ValueError("'obstacles' must be a list (it may be empty)")
    for o in obs:
        if not isinstance(o, dict) or not o.get("name") or not isinstance(o.get("footprint_px"), list) \
                or len(o["footprint_px"]) < 3:
            raise ValueError("each obstacle needs a 'name' and a 'footprint_px' polygon with >= 3 points")
        for pt in o["footprint_px"]:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2
                    and all(isinstance(q, (int, float)) and math.isfinite(q) for q in pt)):
                raise ValueError(f"obstacle '{o['name']}' has a footprint point that is not a finite [x, y]")
        _range(o.get("height_m"), f"height_m of obstacle '{o['name']}'")


def line_of_sight(H, scenario: dict, sigma_cal_m: float, obstacle_shift: dict | None = None) -> dict:
    """Monte Carlo sensitivity analysis. obstacle_shift: {name: (dx_m, dy_m)} applied in world metres (what-if)."""
    validate_scenario(scenario)
    rng = np.random.default_rng(int(scenario.get("seed", 1)))
    n = int(scenario.get("samples", 300))
    sp = float(scenario.get("pixel_pick_sigma_px", 3.0))
    ej = float(scenario.get("eye_jitter_m", 0.25))
    eh, ph = _range(scenario["eye_height_m"], "eye_height_m"), _range(scenario["pedestrian_height_m"], "ped")
    shift = obstacle_shift or {}

    def world(px, sigma_px):
        p = np.asarray(px, float) + rng.normal(0, sigma_px, 2)
        return np.array(apply_h(H, p))

    vis_fracs, states, bad = [], {"Clear": 0, "Partial": 0, "Blocked": 0}, 0
    for _ in range(n):
        eye_xy = world(scenario["eye_ground_px"], sp) + rng.normal(0, sigma_cal_m, 2) + rng.normal(0, ej, 2)
        ped_xy = world(scenario["pedestrian_foot_px"], sp) + rng.normal(0, sigma_cal_m, 2)
        he, hp = rng.uniform(*eh), rng.uniform(*ph)
        prisms = []
        for o in scenario["obstacles"]:
            poly = np.array([world(p, sp) for p in o["footprint_px"]]) + rng.normal(0, sigma_cal_m, 2)
            if o["name"] in shift:
                poly = poly + np.array(shift[o["name"]], float)
            prisms.append((poly, rng.uniform(*_range(o["height_m"], "h"))))
        if any(_pip_vec(np.array([eye_xy[0]]), np.array([eye_xy[1]]), p)[0] and he <= h for p, h in prisms):
            bad += 1                                            # eye inside an obstacle: inconsistent sample
            continue
        visible = sum(
            0 if any(segment_blocked((eye_xy[0], eye_xy[1], he), (ped_xy[0], ped_xy[1], hp * f), p, h)
                     for p, h in prisms) else 1
            for f in BODY_FRACTIONS)
        v = visible / len(BODY_FRACTIONS)
        vis_fracs.append(v)
        states["Clear" if v == 1 else "Blocked" if v == 0 else "Partial"] += 1
    ok = len(vis_fracs)
    if ok == 0:
        return {"geometric_label": "Uncertain", "valid_samples": 0, "eye_inside_obstacle_samples": bad,
                "fractions": {}, "reason": "no valid samples (eye point falls inside an obstacle: check inputs)"}
    fr = {k: round(v / ok, 3) for k, v in states.items()}
    top = max(fr, key=fr.get)
    label = top if fr[top] >= ROBUSTNESS else "Uncertain"
    arr = np.array(vis_fracs)
    return {"geometric_label": label, "fractions": fr, "valid_samples": ok, "eye_inside_obstacle_samples": bad,
            "visible_body_share_mean": round(float(arr.mean()), 3),
            "visible_body_share_p10_p90": [round(float(np.percentile(arr, 10)), 3), round(float(np.percentile(arr, 90)), 3)],
            "robustness_rule": f"label reported only if >= {int(ROBUSTNESS * 100)}% of samples agree, else Uncertain"}


def what_if_setback(H, scenario: dict, sigma_cal_m: float) -> dict | None:
    """HYPOTHETICAL: slide one obstacle along a ground direction by each distance and re-run the analysis."""
    sb = scenario.get("setback")
    if not sb:
        return None
    names = {o["name"] for o in scenario["obstacles"]}
    if sb.get("obstacle") not in names:
        raise ValueError("setback.obstacle must match an obstacle name")
    (x1, y1), (x2, y2) = sb["direction_px"]
    a, b = np.array(apply_h(H, (x1, y1))), np.array(apply_h(H, (x2, y2)))
    d = b - a
    norm = float(np.hypot(*d))
    if norm < 1e-6:
        raise ValueError("setback.direction_px has zero length")
    u = d / norm
    rows = []
    dists = sb.get("distances_m", [0, 1, 2, 3])
    if not isinstance(dists, list) or not dists or not all(
            isinstance(d, (int, float)) and math.isfinite(d) and 0 <= d <= 20 for d in dists):
        raise ValueError("setback.distances_m must be a non-empty list of distances between 0 and 20 metres")
    for dist in dists:
        res = line_of_sight(H, scenario, sigma_cal_m, {sb["obstacle"]: tuple(u * float(dist))})
        rows.append({"obstacle_moved_m": float(dist), "geometric_label": res["geometric_label"],
                     "fractions": res.get("fractions", {}), "visible_body_share_mean": res.get("visible_body_share_mean")})
    return {
        "label": "HYPOTHETICAL",
        "obstacle": sb["obstacle"],
        "direction_world_unit": [round(float(u[0]), 3), round(float(u[1]), 3)],
        "results": rows,
        "held_constant": ["driver eye position and height range", "pedestrian position and height range",
                          "all other obstacles", "the moved obstacle's size and height", "ground-plane calibration"],
        "cannot_predict": ["whether a real vehicle would stop at that spot", "driver or pedestrian behaviour",
                           "driver attention or reaction", "any change in crash risk or frequency"],
    }


def assess(calibration_doc: dict, scenario_doc: dict) -> dict:
    """Full gated analysis. Returns a report dict; without a passing calibration it contains no numbers."""
    gate = check_calibration(calibration_doc)
    report = {"disclaimer": DISCLAIMER, "status": gate["status"],
              "calibration_gate": {k: v for k, v in gate.items() if k != "H"}}
    if gate["status"] != "EXPERIMENTAL_OK":
        report["result"] = "NOT ASSESSABLE: the calibration does not meet the minimum requirements (see reasons)."
        return report
    H, sig = gate["H"], gate["sigma_cal_m"]
    report["measured_by_user"] = {"calibration_source": calibration_doc.get("source"),
                                  "ground_points": calibration_doc.get("ground_points"),
                                  "calibration_error_assumed_m": sig}
    report["user_entered_assumptions"] = {k: scenario_doc.get(k) for k in
                                          ("eye_ground_px", "eye_height_m", "eye_jitter_m", "pedestrian_foot_px",
                                           "pedestrian_height_m", "obstacles", "pixel_pick_sigma_px")}
    report["model_estimate"] = line_of_sight(H, scenario_doc, sig)
    wi = what_if_setback(H, scenario_doc, sig)
    if wi:
        report["hypothetical_what_if"] = wi
    return report


def report_markdown(rep: dict) -> str:
    lines = ["# Line-of-sight check (EXPERIMENTAL)", "", f"> {rep['disclaimer']}", "",
             f"**Status:** {rep['status']}", ""]
    g = rep["calibration_gate"]
    lines += [f"Calibration points: {g.get('n_points')}"]
    if g.get("loo_error_m"):
        lines += [f"Leave-one-out error: mean {g['loo_error_m']['mean']} m, max {g['loo_error_m']['max']} m"]
    if rep["status"] != "EXPERIMENTAL_OK":
        lines += ["", "## Not assessable", ""] + [f"- {r}" for r in g.get("reasons", [])]
        return "\n".join(lines) + "\n"
    m = rep["model_estimate"]
    lines += ["", "## Model estimate (from user-entered assumptions)", "",
              f"- Geometric label: **{m['geometric_label']}** ({m.get('robustness_rule', '')})",
              f"- Sample outcomes: {m.get('fractions')} over {m.get('valid_samples')} valid samples",
              f"- Visible share of pedestrian body: mean {m.get('visible_body_share_mean')}, "
              f"10th-90th percentile {m.get('visible_body_share_p10_p90')}"]
    wi = rep.get("hypothetical_what_if")
    if wi:
        lines += ["", f"## HYPOTHETICAL what-if: moving '{wi['obstacle']}' along the chosen direction", ""]
        lines += [f"- moved {r['obstacle_moved_m']} m: {r['geometric_label']} {r['fractions']}" for r in wi["results"]]
        lines += ["", "Held constant: " + "; ".join(wi["held_constant"]), "",
                  "Cannot predict: " + "; ".join(wi["cannot_predict"])]
    return "\n".join(lines) + "\n"
