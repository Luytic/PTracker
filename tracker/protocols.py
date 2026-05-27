"""Swappable component contracts (Protocol = structural subtyping / DIP).

Pipeline depends on these interfaces, not concrete classes — swap NN, Kalman,
depth, or frame tracker in factory.py without touching TrackingPipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tracker.types import FlowPhase, FrameTrackerResult, KalmanState, PenTipTrackOutput


@runtime_checkable
class Localizer(Protocol):
    """Strategy: NN backend that localizes the tip in a search window."""

    def initialize(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> None: ...

    def track(self, frame: np.ndarray) -> PenTipTrackOutput | None: ...

    def sync_search_center(self, cx: float, cy: float) -> None: ...

    def search_window_roi(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]: ...

    @property
    def last_peak_score(self) -> float: ...

    def maybe_update_neural_template(self, frame: np.ndarray, conf: float) -> None: ...


@runtime_checkable
class MotionFilter(Protocol):
    """Strategy: temporal state estimator (default: constant-velocity Kalman 6D)."""

    def reset(self, x: float, y: float, z: float) -> None: ...

    def predict(self) -> KalmanState: ...

    def update(self, x: float, y: float, z: float, confidence: float = 1.0) -> None: ...

    def sync_velocity(self, vx: float, vy: float, *, blend: float = 1.0) -> None: ...

    def decay_velocity(self, factor: float = 0.9) -> None: ...

    def state(self) -> KalmanState: ...


@runtime_checkable
class DepthModel(Protocol):
    """Strategy: monocular relative depth (default: NN bbox scale vs init ROI)."""

    def reset(self, bbox: tuple[int, int, int, int]) -> None: ...

    def observe_bbox(self, width: float, height: float) -> None: ...

    @property
    def scale_ratio(self) -> float: ...

    @property
    def last_step_factor(self) -> float: ...

    def estimate(self) -> float: ...


@runtime_checkable
class FrameTracker(Protocol):
    """Strategy: per-frame measurements — NN on schedule + LK flow (HybridFrameTracker)."""

    def initialize(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> None: ...

    def estimate_flow(
        self,
        frame: np.ndarray,
        center_xy: tuple[float, float],
    ) -> FlowPhase | None: ...

    def track(
        self,
        frame: np.ndarray,
        pred_xy: tuple[float, float],
        *,
        motion_xy: tuple[float, float] | None = None,
        flow_phase: FlowPhase | None = None,
    ) -> FrameTrackerResult | None: ...

    def display_bbox(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
    ) -> tuple[int, int, int, int]: ...

    @property
    def last_bbox(self) -> tuple[int, int, int, int] | None: ...

    @property
    def last_search_roi(self) -> tuple[int, int, int, int] | None: ...

    @property
    def last_search_radius_px(self) -> int: ...

    @property
    def nn_wh(self) -> tuple[float, float]: ...
