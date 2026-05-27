"""TrackingStateMachine — FSM: visible / uncertain / lost / reacquired."""

from __future__ import annotations

from tracker.config import TrackingConfig


class TrackingStateMachine:
    """Finite state machine; lost streak tied to nn_interval; flow does not reset streak."""

    def __init__(self, config: TrackingConfig) -> None:
        self._cfg = config
        self._lost_streak = 0
        self._bad_peak_streak = 0
        self.state = "visible"
        self._emit_reacquired_next = False

    def reset(self, *, reacquired: bool = False) -> None:
        self._lost_streak = 0
        self._bad_peak_streak = 0
        self.state = "visible"
        self._emit_reacquired_next = reacquired

    def update(
        self,
        *,
        frame_index: int,
        nn_ran: bool,
        nn_peak_score: float,
        has_valid_measurement: bool,
        measure_conf: float,
    ) -> str:
        if self.state == "lost":
            return "lost"

        cfg = self._cfg
        lost_after_no_measure = max(1, int(cfg.nn_interval))

        if frame_index >= cfg.warmup_frames and nn_ran:
            if has_valid_measurement:
                self._bad_peak_streak = 0
            else:
                self._bad_peak_streak += 1

        if has_valid_measurement and measure_conf >= cfg.measure_conf_threshold:
            self._lost_streak = 0
            self.state = (
                "visible" if measure_conf >= cfg.tracking_conf_threshold else "uncertain"
            )
        else:
            self._lost_streak += 1
            if self._lost_streak > lost_after_no_measure:
                self.state = "lost"
            elif self._bad_peak_streak >= cfg.lost_after_bad_peaks:
                self.state = "lost"
            else:
                self.state = "uncertain"

        if (
            frame_index >= cfg.warmup_frames
            and nn_ran
            and nn_peak_score < cfg.peak_reinit_threshold
        ):
            self.state = "lost"

        if self._emit_reacquired_next:
            self._emit_reacquired_next = False
            self.state = "reacquired"

        return self.state
