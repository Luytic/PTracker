"""Factory: compose production stack (Factory + DI wiring point).

Single place to swap PenTipTrack version, Kalman, depth, or frame tracker.
TrackingPipeline itself stays unchanged (Facade over protocols).
"""

from __future__ import annotations

from dataclasses import replace
from tracker.backends.pentiptrack import PenTipTrackLocalizer
from tracker.config import TrackingConfig
from tracker.depth.relative import RelativeDepthEstimator
from tracker.motion.kalman import Kalman3D
from tracker.tracking.frame_tracker import HybridFrameTracker
from tracker.tracking.pipeline import TrackingPipeline


def create_tracking_pipeline(
    fps: float = 30.0,
    *,
    pentiptrack_version: str = "v3",
    nn_interval: int = 3,
    config: TrackingConfig | None = None,
) -> TrackingPipeline:
    """Build TrackingPipeline with default PenTipTrack + LK + Kalman + bbox depth."""
    base = config or TrackingConfig()
    cfg = replace(base, nn_interval=nn_interval)

    localizer = PenTipTrackLocalizer(
        version=pentiptrack_version,
        min_peak_score=cfg.peak_lost_threshold,
    )
    method = f"pentiptrack_{pentiptrack_version.lower()}"
    depth = RelativeDepthEstimator()
    frame_tracker = HybridFrameTracker(
        localizer,
        config=cfg,
        method_name=method,
    )
    motion = Kalman3D(
        dt=1.0 / max(1.0, fps),
        r_xy=cfg.kalman_r_xy,
        r_z=cfg.kalman_r_z,
        q_pos=cfg.kalman_q_pos,
        q_vel=cfg.kalman_q_vel,
    )
    return TrackingPipeline(motion, frame_tracker, depth, config=cfg)


__all__ = ["TrackingPipeline", "create_tracking_pipeline"]
