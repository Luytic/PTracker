from __future__ import annotations

import numpy as np

SCALE_RATIO_MIN = 0.25
SCALE_RATIO_MAX = 4.0
MAX_FRAME_SCALE_DELTA = 0.30
MAX_APPROACH_FRAME_DELTA = 0.45

MIN_FRAME_SCALE_FACTOR = 1.0 - MAX_FRAME_SCALE_DELTA  # 0.7
MAX_FRAME_SCALE_FACTOR = 1.0 + MAX_FRAME_SCALE_DELTA  # 1.3
MAX_APPROACH_FRAME_FACTOR = 1.0 + MAX_APPROACH_FRAME_DELTA  # 1.45


def clip_scale_ratio(ratio: float) -> float:
    return float(np.clip(ratio, SCALE_RATIO_MIN, SCALE_RATIO_MAX))


def limit_scale_step(current: float, target: float) -> float:
    cur = float(current)
    if cur <= 0.0:
        return clip_scale_ratio(target)
    lo = cur * MIN_FRAME_SCALE_FACTOR
    hi = cur * MAX_FRAME_SCALE_FACTOR
    return clip_scale_ratio(float(np.clip(target, lo, hi)))


def limit_approach_scale_step(current: float, target: float) -> float:
    """Faster accumulation when the object moves closer (scale grows)."""
    cur = float(current)
    if cur <= 0.0:
        return clip_scale_ratio(target)
    lo = cur * MIN_FRAME_SCALE_FACTOR
    hi = cur * MAX_APPROACH_FRAME_FACTOR
    return clip_scale_ratio(float(np.clip(target, lo, hi)))
