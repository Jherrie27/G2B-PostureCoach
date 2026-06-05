"""Runtime configuration. Reads env vars with sensible defaults."""
import os
import platform


def is_pi() -> bool:
    return platform.machine() in ("aarch64", "armv7l")


def mediapipe_complexity() -> int:
    v = os.environ.get("G2B_MP_COMPLEXITY")
    if v is not None:
        return int(v)
    return 0 if is_pi() else 1


def target_fps() -> int:
    v = os.environ.get("G2B_TARGET_FPS")
    if v is not None:
        return int(v)
    return 12 if is_pi() else 15


def camera_resolution():
    # Same default on both for now; kept as a hook for per-device tuning.
    return (640, 480)
