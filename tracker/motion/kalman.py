"""Kalman3D — MotionFilter impl: constant-velocity filter, state [x,y,z,vx,vy,vz]."""

from __future__ import annotations

import numpy as np

from tracker.types import KalmanState


class Kalman3D:
    """Implements MotionFilter; R scaled by 1/confidence on update."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        *,
        r_xy: float = 0.05,
        r_z: float = 0.12,
        q_pos: float = 0.05,
        q_vel: float = 0.24,
    ) -> None:
        self.dt = float(dt)
        self._x = np.zeros((6, 1), dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64) * 15.0
        self._F = np.eye(6, dtype=np.float64)
        self._F[0, 3] = self.dt
        self._F[1, 4] = self.dt
        self._F[2, 5] = self.dt
        self._H = np.zeros((3, 6), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._Q = np.diag([q_pos, q_pos, 0.03, q_vel, q_vel, 0.06])
        self._R = np.diag([r_xy, r_xy, r_z])
        self.initialized = False

    def reset(self, x: float, y: float, z: float) -> None:
        self._x[:] = 0.0
        self._x[0, 0] = float(x)
        self._x[1, 0] = float(y)
        self._x[2, 0] = float(z)
        self._P = np.eye(6, dtype=np.float64) * 4.0
        self.initialized = True

    def predict(self) -> KalmanState:
        if not self.initialized:
            return KalmanState(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self.state()

    def update(self, x: float, y: float, z: float, confidence: float = 1.0) -> None:
        if not self.initialized:
            self.reset(x, y, z)
            return
        z_vec = np.array([[x], [y], [z]], dtype=np.float64)
        conf = float(np.clip(confidence, 0.05, 1.0))
        R = self._R / conf
        y_res = z_vec - self._H @ self._x
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y_res
        I = np.eye(6, dtype=np.float64)
        self._P = (I - K @ self._H) @ self._P

    def sync_velocity(self, vx: float, vy: float, *, blend: float = 1.0) -> None:
        """Nudge vx,vy toward measured displacement (post-update, not in standard KF)."""
        if not self.initialized:
            return
        b = float(np.clip(blend, 0.0, 1.0))
        self._x[3, 0] = (1.0 - b) * self._x[3, 0] + b * float(vx)
        self._x[4, 0] = (1.0 - b) * self._x[4, 0] + b * float(vy)

    def decay_velocity(self, factor: float = 0.9) -> None:
        if not self.initialized:
            return
        f = float(np.clip(factor, 0.0, 1.0))
        self._x[3, 0] *= f
        self._x[4, 0] *= f
        self._x[5, 0] *= f

    def state(self) -> KalmanState:
        return KalmanState(
            x=float(self._x[0, 0]),
            y=float(self._x[1, 0]),
            z=float(np.clip(self._x[2, 0], 0.0, 1.0)),
            vx=float(self._x[3, 0]),
            vy=float(self._x[4, 0]),
            vz=float(self._x[5, 0]),
        )
