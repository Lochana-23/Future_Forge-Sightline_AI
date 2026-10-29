# Geometry guide (Stages F and G, experimental)

The line-of-sight check is only meaningful if the image can be tied to real ground distances. This guide says what
you must provide. If you cannot provide it, skip the stage: the tool will refuse, which is the correct outcome.

## What you need

1. **At least 5 ground points** (6 to 8 is better) on the road surface, spread across the area you care about, not
   in a line. For each: its pixel position in the video frame and its real position in metres on a flat local map.
2. **How you measured them** (the `source` field). Examples: distances between crosswalk stripe corners measured on
   a satellite map; a measuring wheel on site.
3. **An honest error estimate** (`measurement_error_m`), per coordinate. If you used a map tool, 0.3 to 0.5 m is
   typical, not 0.05 m.

## How to build the world coordinates

1. Choose a corner of the crosswalk as the origin (0, 0) and one direction as the X axis.
2. Measure distances along and across it for each chosen point to get (X, Y) in metres.
3. In the video frame (use the same frame as `define_zones.py`), read the pixel (x, y) of each point.
4. Copy `config\calibration.example.json` to `config\calibration.json`, fill it in and set `"example_only": false`.

Good points: crosswalk stripe corners, painted-line ends, kerb corners, cone positions you can see on the map.
Bad points: anything not on the ground (roofs, poles, car tops), or points you cannot see on both the map and the video.

## What the gate checks

- 5 or more points, a source, units exactly `metres`, an error estimate between 0 and 1 m.
- Points not (nearly) in a line.
- Leave-one-out reprojection error: mean 0.75 m or less and max 1.5 m or less.
Passing means the points are self-consistent. It does not prove they are correct.

## The scenario (your assumptions for one moment)

`config\scenario.example.json` shows the fields. All of these are assumptions, not measurements:

- where the driver's eyes are on the ground (pixels) and a height range;
- the pedestrian's foot position and a height range;
- each obstacle's ground footprint (pixels) and a height range;
- optional: a direction to slide one obstacle for the what-if, and distances.

## How to read the result

- The result is a **geometric** label from an **assumed** eye point on a flat plane. "Blocked" means the sight line
  passes through the modelled obstacle; it does not mean the driver failed to see anything.
- A label is reported only if at least 80% of the random samples agree. Otherwise it is **Uncertain**, which is
  common and honest.
- The what-if slides one obstacle and re-runs the analysis. It holds everything else fixed and cannot predict
  behaviour, stopping position, attention, or crash outcomes.

## Known simplifications

Flat ground; a single eye point per run (plus jitter); obstacles as straight-sided prisms; pedestrian as vertical
sample points; no lens distortion correction; no motion.
