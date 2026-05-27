"""TrackingPipeline — Facade: one `process()` per frame, delegates to injected parts.

Order: Kalman predict → LK flow → HybridFrameTracker → peak trust → fusion → FSM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tracker.bbox import roi_xywh_to_xyxy
from tracker.config import TrackingConfig
from tracker.protocols import DepthModel, FrameTracker, MotionFilter
from tracker.tracking.fusion import FusionResult, apply_measurement_fusion
from tracker.tracking.peak_trust import PeakTrustTracker
from tracker.tracking.policy.state_machine import TrackingStateMachine
from tracker.types import FlowPhase, FrameTimings, FrameTrackerResult, KalmanState, Measurement, PipelineOutput


@dataclass
class _FrameContext:
    """Per-frame scratch bundle between measure and fuse (not part of public API)."""

    chosen: Measurement
    bbox: tuple[int, int, int, int]
    method: str
    search_roi: tuple[int, int, int, int] | None
    search_radius_px: int
    nn_peak_score: float
    nn_ran: bool
    frame_out: FrameTrackerResult | None


class TrackingPipeline:
    """Orchestrator: predict → measure → fuse → state (Facade over injected components)."""

    def __init__(
        self,
        motion: MotionFilter,
        frame_tracker: FrameTracker,
        depth: DepthModel,
        *,
        config: TrackingConfig | None = None,
    ) -> None:
        self._config = config or TrackingConfig()
        self._motion = motion
        self._frames = frame_tracker
        self._depth = depth
        self._fsm = TrackingStateMachine(self._config)
        self._peak_trust = PeakTrustTracker()
        self.frame_index = 0
        self._last_nn_conf = 0.0

    @property
    def tracking_state(self) -> str:
        return self._fsm.state

    def initialize(
        self,
        frame: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
        *,
        reacquired: bool = False,
    ) -> None:
        bbox = roi_xywh_to_xyxy(roi_xywh)
        self._frames.initialize(frame, bbox)
        self._depth.reset(bbox)
        x, y, w, h = roi_xywh
        self._motion.reset(x + w * 0.5, y + h * 0.5, 0.5)
        self._fsm.reset(reacquired=reacquired)
        self._peak_trust.reset()
        self.frame_index = 0
        self._last_nn_conf = 0.0

    def process(self, frame: np.ndarray) -> PipelineOutput:
        """Run full hybrid pipeline for one BGR frame; returns tip, bbox, state, timings."""
        t0 = time.perf_counter()
        timings = FrameTimings()

        pre, pred, timings = self._predict(timings)
        motion_xy = (pre.x, pre.y)  # pre-update Kalman center — LK ROI anchor
        flow_phase = self._frames.estimate_flow(frame, motion_xy)
        frame_out, timings = self._track_frame(
            frame, pred, pre, motion_xy, flow_phase, timings
        )
        ctx = self._unpack_frame_result(frame_out, pred)

        self._peak_trust.observe_peak(
            nn_ran=ctx.nn_ran, nn_peak_score=ctx.nn_peak_score
        )
        has_valid = self._peak_trust.is_trusted(
            self._config,
            nn_ran=ctx.nn_ran,
            nn_peak_score=ctx.nn_peak_score,
            measurement_valid=ctx.chosen.valid,
            measurement_conf=ctx.chosen.confidence,
        )
        inv_dt = 1.0 / max(1e-6, float(getattr(self._motion, "dt", 1.0 / 30.0)))

        fusion = apply_measurement_fusion(
            cfg=self._config,
            motion=self._motion,
            frames=self._frames,
            depth=self._depth,
            pre=pre,
            chosen=ctx.chosen,
            frame_out=ctx.frame_out,
            has_valid=has_valid,
            nn_peak_score=ctx.nn_peak_score,
            inv_dt=inv_dt,
        )
        timings.kalman_update_ms = fusion.kalman_update_ms

        if ctx.nn_ran:
            self._last_nn_conf = float(ctx.chosen.confidence)

        state = self._fsm.update(
            frame_index=self.frame_index,
            nn_ran=ctx.nn_ran,
            nn_peak_score=ctx.nn_peak_score,
            has_valid_measurement=has_valid,
            measure_conf=float(ctx.chosen.confidence) if ctx.nn_ran else 0.0,
        )

        timings.total_ms = (time.perf_counter() - t0) * 1000
        self.frame_index += 1
        return self._build_output(
            frame,
            ctx,
            fusion,
            state,
            timings,
            flow_phase,
        )

    def _predict(self, timings: FrameTimings) -> tuple[KalmanState, KalmanState, FrameTimings]:
        tp = time.perf_counter()
        pre = self._motion.state()
        pred = self._motion.predict()
        timings.kalman_predict_ms = (time.perf_counter() - tp) * 1000
        return pre, pred, timings

    def _track_frame(
        self,
        frame: np.ndarray,
        pred: KalmanState,
        pre: KalmanState,
        motion_xy: tuple[float, float],
        flow_phase: FlowPhase | None,
        timings: FrameTimings,
    ) -> tuple[FrameTrackerResult | None, FrameTimings]:
        tc = time.perf_counter()
        frame_out = self._frames.track(
            frame,
            (pred.x, pred.y),
            motion_xy=motion_xy,
            flow_phase=flow_phase,
        )
        timings.cv_ms = (time.perf_counter() - tc) * 1000
        timings.nn_ms = float(frame_out.nn_ms) if frame_out is not None else 0.0
        return frame_out, timings

    def _unpack_frame_result(
        self,
        frame_out: FrameTrackerResult | None,
        pred: KalmanState,
    ) -> _FrameContext:
        if frame_out is None:
            chosen = Measurement(pred.x, pred.y, 0.0, "kalman_prediction", valid=False)
            bbox = self._frames.last_bbox or (
                int(pred.x) - 24,
                int(pred.y) - 24,
                int(pred.x) + 24,
                int(pred.y) + 24,
            )
            return _FrameContext(
                chosen=chosen,
                bbox=bbox,
                method="kalman_prediction",
                search_roi=self._frames.last_search_roi,
                search_radius_px=int(self._frames.last_search_radius_px),
                nn_peak_score=0.0,
                nn_ran=False,
                frame_out=None,
            )

        return _FrameContext(
            chosen=frame_out.measurement,
            bbox=frame_out.bbox,
            method=frame_out.method,
            search_roi=frame_out.search_roi,
            search_radius_px=int(frame_out.search_radius_px),
            nn_peak_score=float(frame_out.nn_peak_score),
            nn_ran=bool(frame_out.nn_ran_this_frame),
            frame_out=frame_out,
        )

    def _build_output(
        self,
        frame: np.ndarray,
        ctx: _FrameContext,
        fusion: FusionResult,
        state: str,
        timings: FrameTimings,
        flow_phase: FlowPhase | None,
    ) -> PipelineOutput:
        kf = self._motion.state()
        scale = self._depth.scale_ratio
        bbox = self._frames.display_bbox(frame, kf.x, kf.y)
        frame_out = ctx.frame_out
        flow_dx = float(frame_out.flow_dx) if frame_out is not None else 0.0
        flow_dy = float(frame_out.flow_dy) if frame_out is not None else 0.0
        flow_valid = bool(frame_out.flow_valid) if frame_out is not None else False
        flow_n = int(frame_out.flow_n_points) if frame_out is not None else 0
        flow_roi = frame_out.flow_roi if frame_out is not None else None
        flow_tracks = (
            frame_out.flow_feature_tracks if frame_out is not None else ()
        )
        flow_rejected = (
            frame_out.flow_rejected_tracks if frame_out is not None else ()
        )
        if frame_out is None and flow_phase is not None:
            flow_tracks = flow_phase.feature_tracks
            flow_rejected = flow_phase.rejected_tracks

        return PipelineOutput(
            x=float(kf.x),
            y=float(kf.y),
            z_relative=float(np.clip(fusion.z, 0.0, 1.0)),
            confidence=float(np.clip(self._last_nn_conf, 0.0, 1.0)),  # sticky last NN conf
            tracking_state=state,
            method_used=fusion.method,
            bbox=bbox,
            nn_inference_used=ctx.nn_ran,
            nn_peak_score=ctx.nn_peak_score,
            search_radius_px=ctx.search_radius_px,
            object_scale_ratio=scale,
            search_roi=ctx.search_roi,
            timings=timings,
            flow_dx=flow_dx,
            flow_dy=flow_dy,
            flow_valid=flow_valid,
            flow_n_points=flow_n,
            flow_roi=flow_roi,
            flow_feature_tracks=flow_tracks,
            flow_rejected_tracks=flow_rejected,
            bbox_scale_step=fusion.bbox_scale_delta,
            bbox_scale_updated=fusion.bbox_scale_valid,
        )
