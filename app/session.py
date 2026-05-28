"""WebcamTrackingSession — application loop: capture → pipeline → render → log → metrics."""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.camera import WINDOW_NAME, Webcam, destroy_window
from app.config import AppConfig
from app.recording import VideoRecorder
from app.ui import RoiSelector
from app.telemetry import (
    JsonlLogger,
    RuntimeProfiler,
    draw_depth_trajectory,
    draw_tracking_overlay,
)
from tracker.tracking import TrackingPipeline
from tracker.types import PipelineOutput

_QUIT_KEYS = frozenset({27, ord("q")})
_RESELECT_KEY = ord("r")
_TRAJECTORY_MAX = 3000


class _LoopMetrics:
    """Wall-clock heuristics: estimated dropped frames vs camera nominal FPS."""

    def __init__(self, fps: float) -> None:
        self._frame_step_ms = 1000.0 / max(1.0, fps)
        self.dropped_frames = 0
        self.slow_iterations = 0
        self._prev_ts = 0.0

    def on_frame(self, ts_ms: float, frame_idx: int) -> None:
        if frame_idx <= 0:
            self._prev_ts = ts_ms
            return
        delta = ts_ms - self._prev_ts
        if delta > self._frame_step_ms * 1.5:
            self.dropped_frames += max(0, int(round(delta / self._frame_step_ms)) - 1)
        if delta > self._frame_step_ms * 1.8:
            self.slow_iterations += 1
        self._prev_ts = ts_ms

    def reset_timing(self) -> None:
        self._prev_ts = 0.0


class WebcamTrackingSession:
    """Orchestrates UI + TrackingPipeline; on lost → RoiSelector (re-acquisition)."""

    def __init__(
        self,
        config: AppConfig,
        webcam: Webcam,
        pipeline: TrackingPipeline,
    ) -> None:
        self._config = config
        self._webcam = webcam
        self._pipeline = pipeline
        self._roi = RoiSelector(WINDOW_NAME)
        self._profiler = RuntimeProfiler(window=300)
        self._metrics = _LoopMetrics(webcam.fps)
        self._trajectory: list[tuple[float, float, float]] = []
        self._start = time.perf_counter()
        self._frame_idx = 0
        self._awaiting_reinit = False
        self._frame: np.ndarray | None = None
        self._recorder: VideoRecorder | None = None
        self._logger: JsonlLogger | None = None

    def run(self) -> None:
        first_pick = self._roi.select(self._webcam, setup=True)
        if first_pick is None:
            return

        first, roi = first_pick
        self._pipeline.initialize(first, roi)
        self._frame = first
        h, w = first.shape[:2]

        self._recorder = VideoRecorder(self._config.output, self._webcam.fps, (w, h))
        self._logger = JsonlLogger(self._config.log)

        print(
            f"Camera mode: {w}x{h} @ {self._webcam.fps:.1f} fps | {self._config.tracker_label}"
        )
        try:
            while self._tick():
                pass
        finally:
            self._shutdown()
            self._print_summary()

    def _tick(self) -> bool:
        if self._awaiting_reinit:
            return self._handle_reselect()
        return self._handle_tracking_frame()

    def _handle_reselect(self) -> bool:
        picked = self._roi.select(
            self._webcam,
            prompt="Target lost — click LMB (5% box) or drag ROI | q/ESC quit",
        )
        if picked is None:
            return False
        reframe, rroi = picked
        self._pipeline.initialize(reframe, rroi, reacquired=True)
        self._trajectory.clear()
        self._frame = reframe
        self._frame_idx = 0
        self._awaiting_reinit = False
        self._metrics.reset_timing()
        return True

    def _handle_tracking_frame(self) -> bool:
        assert self._frame is not None
        ts_ms = (time.perf_counter() - self._start) * 1000.0
        out = self._pipeline.process(self._frame)
        self._record_frame(out, ts_ms)

        vis = self._render(self._frame, out)
        assert self._recorder is not None
        self._recorder.write(vis)
        cv2.imshow(WINDOW_NAME, vis)

        self._metrics.on_frame(ts_ms, self._frame_idx)
        self._profiler.add(
            total_ms=out.timings.total_ms,
            timestamp_ms=ts_ms,
            nn_used=out.nn_inference_used,
        )

        if out.tracking_state == "lost":
            self._awaiting_reinit = True
            return self._continue_after_key()

        if not self._continue_after_key(reselect=True):
            return False

        ok, next_frame = self._webcam.read()
        if not ok or next_frame is None:
            return False
        self._frame = next_frame
        self._frame_idx += 1
        if self._config.max_frames > 0 and self._frame_idx >= self._config.max_frames:
            return False
        return True

    def _continue_after_key(self, *, reselect: bool = False) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
        if reselect and key == _RESELECT_KEY:
            self._awaiting_reinit = True
        return True

    def _record_frame(self, out: PipelineOutput, ts_ms: float) -> None:
        self._trajectory.append((out.x, out.y, out.object_scale_ratio))
        if len(self._trajectory) > _TRAJECTORY_MAX:
            self._trajectory = self._trajectory[-_TRAJECTORY_MAX:]

        assert self._logger is not None
        self._logger.write(self._log_row(out, ts_ms))

    def _log_row(self, out: PipelineOutput, ts_ms: float) -> dict:
        """Per-frame JSONL: TZ required fields; timing breakdown only in --debug."""
        row: dict = {
            "frame": self._frame_idx,
            "timestamp_ms": ts_ms,
            "x": out.x,
            "y": out.y,
            "z_relative": out.z_relative,
            "confidence": out.confidence,
            "tracking_state": out.tracking_state,
            "method_used": out.method_used,
            "latency_ms": out.timings.total_ms,
        }
        return row

    def _render(self, frame: np.ndarray, out: PipelineOutput) -> np.ndarray:
        vis = draw_tracking_overlay(
            frame,
            bbox=out.bbox,
            point=(out.x, out.y),
            tracking_state=out.tracking_state,
            debug=False,
            search_roi=out.search_roi,
            flow_roi=out.flow_roi,
            flow_dx=out.flow_dx,
            flow_dy=out.flow_dy,
            flow_valid=out.flow_valid,
            flow_n_points=out.flow_n_points,
            flow_feature_tracks=out.flow_feature_tracks,
            flow_rejected_tracks=out.flow_rejected_tracks,
            confidence=out.confidence,
            method_used=out.method_used,
            object_scale_ratio=out.object_scale_ratio,
            bbox_scale_step=out.bbox_scale_step,
            bbox_scale_updated=out.bbox_scale_updated,
            latency_total_ms=out.timings.total_ms,
            latency_cv_ms=out.timings.cv_ms,
            latency_nn_ms=out.timings.nn_ms,
            latency_kf_ms=out.timings.kalman_predict_ms + out.timings.kalman_update_ms,
        )
        return draw_depth_trajectory(vis, self._trajectory)

    def _shutdown(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
        if self._logger is not None:
            self._logger.close()
        self._webcam.release()
        destroy_window()

    def _print_summary(self) -> None:
        summary = self._profiler.summary()
        print("\n--- Runtime Summary ---")
        print(f"Session FPS (actual, wall clock): {summary['session_fps']:.1f}")
        print(f"Processing FPS (1000 / mean pipeline ms): {summary['processing_fps']:.1f}")
        print(f"Avg latency ms: {summary['avg_latency_ms']:.2f}")
        print(f"P95 latency ms: {summary['p95_latency_ms']:.2f}")
        print(f"P99 latency ms: {summary['p99_latency_ms']:.2f}")
        print(f"Dropped/skipped camera frames (est.): {self._metrics.dropped_frames}")
        print(f"NN inference rate: {summary['nn_inference_rate']:.3f}")
