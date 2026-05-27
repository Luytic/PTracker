"""HybridFrameTracker — Strategy (FrameTracker): NN schedule + LK + localizer bridge.

Does not run Kalman; returns raw Measurement for pipeline fusion.
"""

from __future__ import annotations

import time

import numpy as np

from tracker.bbox import bbox_from_center, clamp_bbox_xyxy
from tracker.config import TrackingConfig
from tracker.cv.local_flow import FlowResult, LocalFlowEstimator
from tracker.protocols import Localizer
from tracker.tracking.policy.scheduler import NnScheduler
from tracker.types import FlowPhase, FrameTrackerResult, Measurement


class HybridFrameTracker:
    """FrameTracker impl: NnScheduler gates PenTipTrack; LK shifts NN search center."""

    def __init__(
        self,
        localizer: Localizer,
        *,
        config: TrackingConfig,
        method_name: str = "pentiptrack",
    ) -> None:
        self._localizer = localizer
        self._config = config
        self._method = method_name
        self._scheduler = NnScheduler(config)
        self._flow = LocalFlowEstimator(
            max_corners=config.flow_max_corners,
            min_valid_points=config.flow_min_valid_points,
            max_displacement=config.flow_max_displacement,
            outlier_mad_k=config.flow_outlier_mad_k,
            outlier_min_px=config.flow_outlier_min_px,
        )
        self._last_peak_score = 0.0
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_search_roi: tuple[int, int, int, int] | None = None
        self._last_search_radius_px: int = 24
        self._nn_wh: tuple[float, float] = (0.0, 0.0)

    def initialize(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> None:
        bbox = clamp_bbox_xyxy(bbox_xyxy, frame.shape)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        self._last_bbox = bbox
        self._nn_wh = (float(x2 - x1), float(y2 - y1))
        self._localizer.initialize(frame, bbox)
        self._scheduler.reset()
        self._last_peak_score = float(self._localizer.last_peak_score)
        if self._config.flow_enabled:
            self._flow.reset(frame, (cx, cy), bbox_wh=self._nn_wh)
        self._refresh_search_roi(frame)

    def estimate_flow(
        self,
        frame: np.ndarray,
        center_xy: tuple[float, float],
    ) -> FlowPhase | None:
        if not self._config.flow_enabled or self._last_bbox is None:
            return None
        flow = self._flow.estimate(frame, center_xy, bbox_wh=self._nn_wh)
        return _flow_to_phase(flow)

    def track(
        self,
        frame: np.ndarray,
        pred_xy: tuple[float, float],
        *,
        motion_xy: tuple[float, float] | None = None,
        flow_phase: FlowPhase | None = None,
    ) -> FrameTrackerResult | None:
        if self._last_bbox is None:
            return None

        self._scheduler.on_frame_start()
        run_nn = self._scheduler.should_run(self._last_peak_score)

        flow_center = motion_xy if motion_xy is not None else pred_xy
        if flow_phase is None and self._config.flow_enabled:
            flow_phase = self.estimate_flow(frame, flow_center)

        flow_dx = 0.0
        flow_dy = 0.0
        flow_valid = False
        flow_n = 0
        flow_roi = None
        flow_feature_tracks: tuple[tuple[float, float, float, float], ...] = ()
        flow_rejected_tracks: tuple[tuple[float, float, float, float], ...] = ()
        search_x, search_y = float(pred_xy[0]), float(pred_xy[1])

        if flow_phase is not None:
            flow_dx = float(flow_phase.dx)
            flow_dy = float(flow_phase.dy)
            flow_valid = bool(flow_phase.valid)
            flow_n = int(flow_phase.n_points)
            flow_roi = flow_phase.roi
            flow_feature_tracks = flow_phase.feature_tracks
            flow_rejected_tracks = flow_phase.rejected_tracks
            if flow_valid:
                blend = float(np.clip(self._config.flow_pred_blend, 0.0, 1.5))
                search_x = float(flow_center[0] + flow_dx * blend)
                search_y = float(flow_center[1] + flow_dy * blend)

        self._localizer.sync_search_center(search_x, search_y)
        self._refresh_search_roi(frame)

        nn_out = None
        nn_ms = 0.0
        if run_nn:
            t_nn = time.perf_counter()
            nn_out = self._run_localizer(frame)
            nn_ms = (time.perf_counter() - t_nn) * 1000.0
            self._refresh_search_roi(frame)

        if nn_out is not None:
            x, y, conf = nn_out[0], nn_out[1], nn_out[2]
            self._nn_wh = (float(nn_out[3]), float(nn_out[4]))
            method = self._method
            valid = (
                conf >= self._config.tracking_conf_threshold
                and self._last_peak_score >= self._config.peak_reinit_threshold
            )
            if self._config.flow_enabled and valid:
                self._flow.reset(frame, (x, y), bbox_wh=self._nn_wh)
                flow_feature_tracks = self._flow.last_result.feature_tracks
                flow_rejected_tracks = self._flow.last_result.rejected_tracks
        else:
            x, y = float(pred_xy[0]), float(pred_xy[1])
            conf = 0.0
            method = "kalman_prediction"
            valid = False

        raw_bbox = bbox_from_center(
            frame, x, y, *_bbox_wh(frame, self._nn_wh)
        )
        self._localizer.sync_search_center(x, y)
        self._refresh_search_roi(frame)
        if run_nn and nn_out is not None and valid:
            self._localizer.maybe_update_neural_template(frame, conf)
        self._last_bbox = raw_bbox
        return FrameTrackerResult(
            measurement=Measurement(
                x=float(x),
                y=float(y),
                confidence=float(conf),
                method_used=method,
                valid=valid,
            ),
            bbox=raw_bbox,
            method=method,
            search_roi=self._last_search_roi,
            search_radius_px=int(self._last_search_radius_px),
            nn_peak_score=float(self._last_peak_score),
            nn_ran_this_frame=bool(run_nn),
            nn_ms=float(nn_ms),
            flow_dx=flow_dx,
            flow_dy=flow_dy,
            flow_valid=flow_valid,
            flow_n_points=flow_n,
            flow_roi=flow_roi,
            flow_feature_tracks=flow_feature_tracks,
            flow_rejected_tracks=flow_rejected_tracks,
        )

    def display_bbox(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
    ) -> tuple[int, int, int, int]:
        """Output bbox: center from Kalman (cx,cy), size from last NN wh only."""
        w, h = _bbox_wh(frame, self._nn_wh)
        return bbox_from_center(frame, float(cx), float(cy), w, h)

    def _refresh_search_roi(self, frame: np.ndarray) -> None:
        roi = self._localizer.search_window_roi(frame.shape)
        self._last_search_roi = roi
        x1, y1, x2, y2 = roi
        half_w = max(1, (x2 - x1) // 2)
        half_h = max(1, (y2 - y1) // 2)
        self._last_search_radius_px = int(max(half_w, half_h))

    def _run_localizer(
        self, frame: np.ndarray
    ) -> tuple[float, float, float, float, float] | None:
        out = self._localizer.track(frame)
        if out is None:
            self._last_peak_score = float(self._localizer.last_peak_score)
            return None
        self._last_peak_score = float(out.peak_score)
        return (
            float(out.x),
            float(out.y),
            float(out.confidence),
            float(out.width),
            float(out.height),
        )

    @property
    def nn_wh(self) -> tuple[float, float]:
        return self._nn_wh

    @property
    def last_bbox(self) -> tuple[int, int, int, int] | None:
        return self._last_bbox

    @property
    def last_search_roi(self) -> tuple[int, int, int, int] | None:
        return self._last_search_roi

    @property
    def last_search_radius_px(self) -> int:
        return int(self._last_search_radius_px)


def _bbox_wh(
    frame: np.ndarray,
    wh: tuple[float, float],
) -> tuple[int, int]:
    iw, ih = wh
    if iw <= 0.0 or ih <= 0.0:
        return 24, 24
    w = int(np.clip(round(iw), 8, frame.shape[1] - 1))
    h = int(np.clip(round(ih), 8, frame.shape[0] - 1))
    return w, h


def _flow_to_phase(flow: FlowResult) -> FlowPhase:
    return FlowPhase(
        dx=float(flow.dx),
        dy=float(flow.dy),
        valid=bool(flow.valid),
        n_points=int(flow.n_points),
        roi=flow.roi,
        feature_tracks=flow.feature_tracks,
        rejected_tracks=flow.rejected_tracks,
    )
