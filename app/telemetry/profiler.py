from __future__ import annotations

from collections import deque

import numpy as np


class RuntimeProfiler:
    def __init__(self, window: int = 300) -> None:
        self._lat = deque(maxlen=window)
        self._frame_timestamps = deque(maxlen=window)
        self.nn_used = 0
        self.frames = 0
        self._session_first_ts: float | None = None
        self._session_last_ts = 0.0

    def add(self, total_ms: float, timestamp_ms: float, nn_used: bool) -> None:
        ts = float(timestamp_ms)
        if self._session_first_ts is None:
            self._session_first_ts = ts
        self._session_last_ts = ts
        self._lat.append(float(total_ms))
        self._frame_timestamps.append(ts)
        self.frames += 1
        if nn_used:
            self.nn_used += 1

    def _session_fps(self) -> float:
        if self.frames < 2 or self._session_first_ts is None:
            return 0.0
        dt_ms = self._session_last_ts - self._session_first_ts
        if dt_ms <= 1e-6:
            return 0.0
        return 1000.0 * (self.frames - 1) / dt_ms

    def summary(self) -> dict[str, float]:
        if not self._lat:
            return {
                "session_fps": 0.0,
                "processing_fps": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "nn_inference_rate": 0.0,
            }
        lat = np.asarray(self._lat, dtype=np.float64)
        proc_fps = 1000.0 / max(1e-6, float(np.mean(lat)))
        return {
            "session_fps": float(self._session_fps()),
            "processing_fps": float(proc_fps),
            "avg_latency_ms": float(np.mean(lat)),
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
            "nn_inference_rate": float(self.nn_used / max(1, self.frames)),
        }
