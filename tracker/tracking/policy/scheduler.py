"""NnScheduler — Strategy: when to run expensive NN (interval + low-peak trigger)."""

from __future__ import annotations

from tracker.config import TrackingConfig


class NnScheduler:
    """Returns True on frame 1, every N frames, or when peak < nn_force_peak."""

    def __init__(self, config: TrackingConfig) -> None:
        self._interval = max(1, int(config.nn_interval))
        self._force_peak = float(config.nn_force_peak)
        self._frame_index = 0

    def reset(self) -> None:
        self._frame_index = 0

    def on_frame_start(self) -> None:
        self._frame_index += 1

    def should_run(self, last_peak_score: float) -> bool:
        return (
            self._frame_index == 1
            or self._frame_index % self._interval == 0
            or last_peak_score < self._force_peak
        )
