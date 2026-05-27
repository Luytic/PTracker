"""PeakTrustTracker — gating helper: is this NN frame trusted for fusion + FSM visible?"""

from __future__ import annotations

from tracker.config import TrackingConfig


class PeakTrustTracker:
    """Peak EMA + thresholds; separate from Measurement.valid (localizer-level)."""

    def __init__(self) -> None:
        self._peak_ema = 0.0

    def reset(self) -> None:
        self._peak_ema = 0.0

    @property
    def peak_ema(self) -> float:
        return self._peak_ema

    def observe_peak(self, *, nn_ran: bool, nn_peak_score: float) -> None:
        if not nn_ran or nn_peak_score <= 0.0:
            return
        if self._peak_ema <= 0.0:
            self._peak_ema = nn_peak_score
        else:
            self._peak_ema = max(
                self._peak_ema * 0.9 + nn_peak_score * 0.1,
                nn_peak_score,
            )

    def is_trusted(
        self,
        cfg: TrackingConfig,
        *,
        nn_ran: bool,
        nn_peak_score: float,
        measurement_valid: bool,
        measurement_conf: float,
    ) -> bool:
        peak_trusted = (
            nn_ran
            and nn_peak_score >= cfg.peak_reinit_threshold
            and measurement_conf >= cfg.tracking_conf_threshold
        )
        if peak_trusted and self._peak_ema > 0.0:
            peak_trusted = nn_peak_score >= self._peak_ema * cfg.peak_drop_ratio
        return measurement_valid and peak_trusted
