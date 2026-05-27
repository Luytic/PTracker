"""TrackingConfig — frozen value object; thresholds for FSM, NN, flow, Kalman."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    """Immutable tuning bag; override via factory.replace() or custom instance."""

    warmup_frames: int = 8
    peak_lost_threshold: float = 0.45
    peak_reinit_threshold: float = 0.78
    peak_drop_ratio: float = 0.95
    lost_after_bad_peaks: int = 2
    measure_conf_threshold: float = 0.30
    tracking_conf_threshold: float = 0.58
    nn_interval: int = 3
    nn_force_peak: float = 0.50
    flow_enabled: bool = True
    flow_pred_blend: float = 1.0
    flow_max_corners: int = 48
    flow_min_valid_points: int = 6
    flow_max_displacement: float = 18.0
    flow_outlier_mad_k: float = 2.8
    flow_outlier_min_px: float = 2.5
    kalman_r_xy: float = 0.05
    kalman_r_z: float = 0.12
    kalman_q_pos: float = 0.05
    kalman_q_vel: float = 0.24
    kalman_nn_conf_floor: float = 0.90
    kalman_flow_conf: float = 0.58
    kalman_vel_stop_px: float = 1.25
    kalman_vel_nudge_blend: float = 0.72
    kalman_vel_decay: float = 0.90
