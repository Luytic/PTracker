"""RelativeDepthEstimator — DepthModel impl: scale from NN bbox vs init ROI."""

from __future__ import annotations

import numpy as np

from tracker.depth.scale_limits import (
    clip_scale_ratio,
    limit_approach_scale_step,
    limit_scale_step,
)


class RelativeDepthEstimator:
    """observe_bbox on trusted NN only; estimate() maps scale_ratio → z_relative ∈ [0,1]."""

    def __init__(self) -> None:
        self._scale = 1.0
        self._init_wh: tuple[float, float] = (0.0, 0.0)
        self._last_step_factor = 1.0

    def reset(self, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        self._init_wh = (float(max(1, x2 - x1)), float(max(1, y2 - y1)))
        self._scale = 1.0
        self._last_step_factor = 1.0

    @property
    def scale_ratio(self) -> float:
        return clip_scale_ratio(self._scale)

    @property
    def last_step_factor(self) -> float:
        return float(self._last_step_factor)

    def observe_bbox(self, width: float, height: float) -> None:
        iw, ih = self._init_wh
        w, h = float(width), float(height)
        if iw <= 0.0 or ih <= 0.0 or w <= 0.0 or h <= 0.0:
            return
        target = float(np.sqrt((w / iw) * (h / ih)))
        target = clip_scale_ratio(target)
        start = self._scale
        if target >= start:
            self._scale = limit_approach_scale_step(start, target)
        else:
            self._scale = limit_scale_step(start, target)
        self._last_step_factor = self._scale / max(start, 1e-6)

    def estimate(self) -> float:
        ratio = self.scale_ratio
        return float(max(0.0, min(1.0, 0.5 + (ratio - 1.0) * 0.5)))
