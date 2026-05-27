"""Shared DTOs — data carriers between pipeline stages (no business logic)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Measurement:
    x: float
    y: float
    confidence: float
    method_used: str
    valid: bool = True


@dataclass
class KalmanState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


@dataclass
class FrameTimings:
    kalman_predict_ms: float = 0.0
    cv_ms: float = 0.0
    nn_ms: float = 0.0
    kalman_update_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PenTipTrackOutput:
    x: float
    y: float
    confidence: float
    peak_score: float
    width: float
    height: float


@dataclass(frozen=True)
class FlowPhase:
    dx: float
    dy: float
    valid: bool
    n_points: int
    roi: tuple[int, int, int, int]
    feature_tracks: tuple[tuple[float, float, float, float], ...] = ()
    rejected_tracks: tuple[tuple[float, float, float, float], ...] = ()


@dataclass
class FrameTrackerResult:
    measurement: Measurement
    bbox: tuple[int, int, int, int]
    method: str
    search_roi: tuple[int, int, int, int] | None
    search_radius_px: int
    nn_peak_score: float = 0.0
    nn_ran_this_frame: bool = False
    nn_ms: float = 0.0
    flow_dx: float = 0.0
    flow_dy: float = 0.0
    flow_valid: bool = False
    flow_n_points: int = 0
    flow_roi: tuple[int, int, int, int] | None = None
    flow_feature_tracks: tuple[tuple[float, float, float, float], ...] = ()
    flow_rejected_tracks: tuple[tuple[float, float, float, float], ...] = ()


@dataclass
class PipelineOutput:
    x: float
    y: float
    z_relative: float
    confidence: float
    tracking_state: str
    method_used: str
    bbox: tuple[int, int, int, int]
    nn_inference_used: bool
    nn_peak_score: float
    search_radius_px: int
    object_scale_ratio: float
    search_roi: tuple[int, int, int, int] | None
    timings: FrameTimings
    flow_dx: float = 0.0
    flow_dy: float = 0.0
    flow_valid: bool = False
    flow_n_points: int = 0
    flow_roi: tuple[int, int, int, int] | None = None
    flow_feature_tracks: tuple[tuple[float, float, float, float], ...] = ()
    flow_rejected_tracks: tuple[tuple[float, float, float, float], ...] = ()
    bbox_scale_step: float = 1.0
    bbox_scale_updated: bool = False
