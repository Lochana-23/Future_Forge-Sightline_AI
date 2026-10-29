"""Stage A: environment + video feasibility audit. Read-only: never modifies the video."""
import argparse, ctypes, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 1e9, 1)
    except Exception:
        return None


def audit_env():
    import cv2, numpy
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "nvidia_gpu": run(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else "none found",
        "ffprobe_available": shutil.which("ffprobe") is not None,
    }


def ffprobe(path):
    if not shutil.which("ffprobe"):
        return None
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
               "-of", "json", str(path)])
    try:
        return json.loads(out)["streams"][0]
    except Exception:
        return {"raw": out}


def audit_video(path, outdir, full):
    import cv2, numpy as np
    p = Path(path)
    rep = {"file": str(p), "warnings": []}
    if not p.is_file():
        rep["error"] = "file not found"
        return rep
    rep["size_mb"] = round(p.stat().st_size / 1e6, 1)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        rep["error"] = "OpenCV could not open the file (codec problem?)"
        return rep
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    rep.update(opencv_fps=fps, opencv_frame_count=n, width=w, height=h,
               fourcc="".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)),
               duration_s_from_metadata=round(n / fps, 2) if fps else None)
    rep["ffprobe"] = ffprobe(p)
    if (w, h) != (1200, 1100):
        rep["warnings"].append(f"Resolution is {w}x{h}, not 1200x1100.")
    if not fps or abs(fps - 30) > 0.5:
        rep["warnings"].append(f"FPS reported as {fps}, not 30.")

    # Seek test + sample frames
    sdir = Path(outdir) / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    samples = []
    for frac in (0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
        idx = min(int(n * frac), max(n - 2, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        entry = {"requested_frame": idx, "decoded": bool(ok)}
        if ok:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            entry.update(pos_after_read=int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                         mean_brightness=round(float(g.mean()), 1),
                         sharpness_laplacian_var=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
                         file=f"frame_{idx:06d}.jpg")
            cv2.imwrite(str(sdir / entry["file"]), frame)
            if entry["mean_brightness"] < 60:
                rep["warnings"].append(f"Frame {idx} is dark (mean {entry['mean_brightness']}).")
        else:
            rep["warnings"].append(f"Could not decode frame {idx}.")
        samples.append(entry)
    rep["samples"] = samples

    # Sequential decode speed (first 300 frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t0, got = time.time(), 0
    while got < 300 and cap.read()[0]:
        got += 1
    dt = time.time() - t0
    rep["decode_first_300_frames"] = {"frames_read": got, "seconds": round(dt, 2),
                                      "decode_fps": round(got / dt, 1) if dt else None}

    # Optional full decode: count readable frames + crude frame-to-frame change stats
    if full:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        read, prev, diffs = 0, None, []
        t0 = time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            read += 1
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (150, 137))
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(g, prev))))
            prev = g
        d = np.array(diffs) if diffs else np.zeros(1)
        top = np.argsort(d)[-5:][::-1]
        rep["full_decode"] = {
            "frames_read": read, "seconds": round(time.time() - t0, 1),
            "matches_metadata_count": read == n,
            "median_frame_diff": round(float(np.median(d)), 3),
            "p99_frame_diff": round(float(np.percentile(d, 99)), 3),
            "largest_jumps(frame,time_s,diff)": [(int(i) + 1, round((int(i) + 1) / fps, 2) if fps else None,
                                                  round(float(d[i]), 2)) for i in top],
            "note": "Large jumps are only hints (cut, camera bump, dropped frames, lighting); not proof.",
        }
        if read != n:
            rep["warnings"].append(f"Readable frames ({read}) != metadata count ({n}).")
    cap.release()
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="outputs/audit")
    ap.add_argument("--full", action="store_true", help="decode every frame once (1-3 min)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    report = {"environment": audit_env(), "video": audit_video(a.video, a.out, a.full)}
    (Path(a.out) / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nSaved:", Path(a.out) / "audit_report.json")
