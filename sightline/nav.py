"""Pure navigation arithmetic (no UI), so it can be unit-tested."""


def clamp_frame(frame: int, frame_count: int) -> int:
    return max(0, min(int(frame), frame_count - 1))


def step_by_seconds(frame: int, seconds: float, fps: float, frame_count: int) -> int:
    """Move by a number of seconds (negative = back), clamped to the video."""
    return clamp_frame(frame + int(round(seconds * fps)), frame_count)


def step_by_frames(frame: int, n: int, frame_count: int) -> int:
    return clamp_frame(frame + n, frame_count)
