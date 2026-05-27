"""Measurement fusion — Policy: priority NN > optical flow > Kalman predict-only.

Pure function (no hidden state); pipeline passes MotionFilter/DepthModel in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tracker.config import TrackingConfig
from tracker.protocols import DepthModel, FrameTracker, MotionFilter
from tracker.types import FrameTrackerResult, KalmanState, Measurement


@dataclass
class FusionResult:
    method: str
    z: float
    kalman_update_ms: float
    bbox_scale_delta: float
    bbox_scale_valid: bool


def apply_measurement_fusion(
    *,
    cfg: TrackingConfig,
    motion: MotionFilter,
    frames: FrameTracker,
    depth: DepthModel,
    pre: KalmanState,
    chosen: Measurement,
    frame_out: FrameTrackerResult | None,
    has_valid: bool,
    nn_peak_score: float,
    inv_dt: float,
) -> FusionResult:
    """Apply fusion rules; mutates motion/depth in place; returns method + z for output."""
    method = "kalman_prediction"
    z = depth.estimate()
    kalman_update_ms = 0.0
    bbox_scale_delta = 1.0
    bbox_scale_valid = False

    if has_valid:
        tu = time.perf_counter()
        w, h = frames.nn_wh
        depth.observe_bbox(w, h)
        bbox_scale_delta = depth.last_step_factor
        bbox_scale_valid = True
        z = depth.estimate()
        kf_conf = max(float(chosen.confidence), cfg.kalman_nn_conf_floor)
        motion.update(chosen.x, chosen.y, z, confidence=kf_conf)
        motion.sync_velocity(
            (float(chosen.x) - pre.x) * inv_dt,
            (float(chosen.y) - pre.y) * inv_dt,
            blend=cfg.kalman_vel_nudge_blend,
        )
        kalman_update_ms = (time.perf_counter() - tu) * 1000
        method = chosen.method_used
    elif (
        frame_out is not None
        and cfg.flow_enabled
        and frame_out.flow_valid
        and nn_peak_score >= cfg.peak_lost_threshold
        and cfg.kalman_flow_conf >= cfg.measure_conf_threshold
    ):
        fx = float(pre.x + frame_out.flow_dx)
        fy = float(pre.y + frame_out.flow_dy)
        z = depth.estimate()
        tu = time.perf_counter()
        motion.update(fx, fy, z, confidence=cfg.kalman_flow_conf)
        flow_mag = float(np.hypot(frame_out.flow_dx, frame_out.flow_dy))
        vel_blend = (
            1.0
            if flow_mag <= cfg.kalman_vel_stop_px
            else cfg.kalman_vel_nudge_blend
        )
        motion.sync_velocity(
            float(frame_out.flow_dx) * inv_dt,
            float(frame_out.flow_dy) * inv_dt,
            blend=vel_blend,
        )
        kalman_update_ms = (time.perf_counter() - tu) * 1000
        method = "optical_flow"
    else:
        z = depth.estimate()
        method = "kalman_prediction"
        motion.decay_velocity(cfg.kalman_vel_decay)

    return FusionResult(
        method=method,
        z=z,
        kalman_update_ms=kalman_update_ms,
        bbox_scale_delta=bbox_scale_delta,
        bbox_scale_valid=bbox_scale_valid,
    )
