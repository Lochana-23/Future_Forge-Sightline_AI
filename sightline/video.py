"""Video loading, seeking and frame export (Stage B, extended in C2)."""
from __future__ import annotations

import threading
from pathlib import Path

import cv2


class VideoError(Exception):
    """Raised for unreadable or unusable video files."""


class VideoSource:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoError(f"File not found: {self.path}")
        self._lock = threading.RLock()   # one cv2.VideoCapture is shared by all browser sessions: serialise seek+read
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise VideoError("OpenCV could not open this file (unsupported codec?).")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_count <= 0 or self.fps <= 0:
            self.cap.release()
            raise VideoError("Video reports no frames or an invalid frame rate.")

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def time_of(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def frame_of(self, time_s: float) -> int:
        return self.clamp(int(round(time_s * self.fps)))

    def clamp(self, frame_idx: int) -> int:
        return max(0, min(int(frame_idx), self.frame_count - 1))

    def read_bgr(self, frame_idx: int):
        """Return the BGR frame at frame_idx (clamped). Raises VideoError if unreadable."""
        idx = self.clamp(frame_idx)
        with self._lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise VideoError(f"Could not decode frame {idx}.")
        return frame

    def read_rgb(self, frame_idx: int):
        return cv2.cvtColor(self.read_bgr(frame_idx), cv2.COLOR_BGR2RGB)

    def iter_frames(self, indices):
        """Yield (index, frame_or_None) for the wanted indices in ascending order.

        Seeks once, then decodes sequentially and skips unwanted frames with grab().
        Much faster than seeking per frame on inter-frame codecs such as MPEG-1.
        frame is None if a wanted frame could not be decoded. Stops at end of stream.
        """
        wanted = sorted({self.clamp(i) for i in indices})
        if not wanted:
            return
        with self._lock:
            want = set(wanted)
            pos = wanted[0]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            while pos <= wanted[-1]:
                if pos in want:
                    ok, frame = self.cap.read()
                    yield pos, (frame if ok and frame is not None else None)
                    if not ok:
                        return
                elif not self.cap.grab():
                    return
                pos += 1

    def save_frame(self, frame_idx: int, dest) -> Path:
        """Write the exact frame as a JPEG at dest (via a temp file, so a failure leaves no partial image)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame = self.read_bgr(frame_idx)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise VideoError(f"Could not encode frame {frame_idx}.")
        from .fileio import atomic_write_bytes
        try:
            atomic_write_bytes(dest, buf.tobytes())
        except OSError as e:
            raise VideoError(f"Could not write {dest}: {e}") from e
        return dest

    def export_frame(self, frame_idx: int, out_dir) -> Path:
        """Save frame_NNNNNN.jpg in out_dir (a plain frame export, not tied to any event); returns the path."""
        idx = self.clamp(frame_idx)
        return self.save_frame(idx, Path(out_dir) / f"frame_{idx:06d}.jpg")

    def close(self):
        with self._lock:
            self.cap.release()
