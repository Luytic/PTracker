"""LocalFlowEstimator — classical CV: pyramidal LK + MAD outlier rejection on displacement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowResult:
    dx: float
    dy: float
    valid: bool
    n_points: int
    roi: tuple[int, int, int, int]
    feature_tracks: tuple[tuple[float, float, float, float], ...] = ()
    rejected_tracks: tuple[tuple[float, float, float, float], ...] = ()


class LocalFlowEstimator:
    """Pyramidal LK flow — robust median displacement; ROI matches NN bbox size."""

    def __init__(
        self,
        *,
        max_corners: int = 48,
        quality_level: float = 0.015,
        min_distance: float = 4.0,
        min_valid_points: int = 6,
        max_displacement: float = 18.0,
        outlier_mad_k: float = 2.8,
        outlier_min_px: float = 2.5,
        lk_win_size: int = 15,
        lk_max_level: int = 3,
        default_bbox_wh: tuple[float, float] = (32.0, 32.0),
    ) -> None:
        self._default_bbox_wh = (
            max(8.0, float(default_bbox_wh[0])),
            max(8.0, float(default_bbox_wh[1])),
        )
        self._max_corners = max(8, int(max_corners))
        self._quality_level = float(quality_level)
        self._min_distance = float(min_distance)
        self._min_valid_points = max(3, int(min_valid_points))
        self._max_displacement = float(max_displacement)
        self._outlier_mad_k = float(outlier_mad_k)
        self._outlier_min_px = float(outlier_min_px)
        self._lk_win = (max(5, int(lk_win_size)), max(5, int(lk_win_size)))
        self._lk_max_level = max(0, int(lk_max_level))
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._last_result = FlowResult(0.0, 0.0, False, 0, (0, 0, 0, 0))

    @property
    def last_result(self) -> FlowResult:
        return self._last_result

    def reset(
        self,
        frame: np.ndarray,
        center_xy: tuple[float, float],
        *,
        bbox_wh: tuple[float, float] | None = None,
    ) -> None:
        wh = self._default_bbox_wh if bbox_wh is None else bbox_wh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = _roi_from_bbox(frame.shape, center_xy, wh)
        pts = self._detect_corners(gray, roi)
        self._prev_gray = gray
        self._prev_pts = pts
        tracks = _points_as_dots(pts)
        self._last_result = FlowResult(0.0, 0.0, False, len(tracks), roi, tracks)

    def estimate(
        self,
        frame: np.ndarray,
        center_xy: tuple[float, float],
        *,
        bbox_wh: tuple[float, float],
    ) -> FlowResult:
        wh = bbox_wh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = _roi_from_bbox(frame.shape, center_xy, wh)

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 3:
            self.reset(frame, center_xy, bbox_wh=wh)
            self._last_result = FlowResult(0.0, 0.0, False, 0, roi)
            return self._last_result

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_pts,
            None,
            winSize=self._lk_win,
            maxLevel=self._lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        if next_pts is None or status is None:
            self.reset(frame, center_xy, bbox_wh=wh)
            self._last_result = FlowResult(0.0, 0.0, False, 0, roi)
            return self._last_result

        partition = _partition_flow_tracks(
            self._prev_pts,
            next_pts,
            status,
            mad_k=self._outlier_mad_k,
            min_floor_px=self._outlier_min_px,
            min_valid_points=self._min_valid_points,
        )

        if partition.n_inliers < self._min_valid_points:
            self.reset(frame, center_xy, bbox_wh=wh)
            self._last_result = FlowResult(
                0.0,
                0.0,
                False,
                partition.n_inliers,
                roi,
                partition.used_tracks,
                partition.rejected_tracks,
            )
            return self._last_result

        prev_in = partition.inlier_prev
        next_in = partition.inlier_next
        disp = next_in - prev_in
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))
        mag = float(np.hypot(dx, dy))
        if mag > self._max_displacement:
            self.reset(frame, center_xy, bbox_wh=wh)
            self._last_result = FlowResult(
                dx,
                dy,
                False,
                partition.n_inliers,
                roi,
                partition.used_tracks,
                partition.rejected_tracks,
            )
            return self._last_result

        self._prev_gray = gray
        self._prev_pts = next_in.reshape(-1, 1, 2).astype(np.float32)
        if len(self._prev_pts) < self._min_valid_points:
            self._append_corners(gray, roi)

        result = FlowResult(
            dx,
            dy,
            True,
            partition.n_inliers,
            roi,
            partition.used_tracks,
            partition.rejected_tracks,
        )
        self._last_result = result
        return result

    def _detect_corners(
        self,
        gray: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = roi
        patch = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(
            patch,
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
        )
        if pts is None:
            return None
        pts = pts.reshape(-1, 1, 2).astype(np.float32)
        pts[:, 0, 0] += float(x1)
        pts[:, 0, 1] += float(y1)
        return pts

    def _append_corners(self, gray: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        if self._prev_pts is None:
            return
        fresh = self._detect_corners(gray, roi)
        if fresh is None or len(fresh) == 0:
            return
        existing = self._prev_pts.reshape(-1, 2)
        merged_pts: list[np.ndarray] = [existing[i] for i in range(len(existing))]
        min_dist = max(3.0, self._min_distance)
        for i in range(len(fresh)):
            p = fresh[i, 0]
            if existing.size and np.min(np.linalg.norm(existing - p, axis=1)) < min_dist:
                continue
            merged_pts.append(p)
            if len(merged_pts) >= self._max_corners:
                break
        self._prev_pts = np.asarray(merged_pts, dtype=np.float32).reshape(-1, 1, 2)


@dataclass(frozen=True)
class _FlowPartition:
    inlier_prev: np.ndarray
    inlier_next: np.ndarray
    used_tracks: tuple[tuple[float, float, float, float], ...]
    rejected_tracks: tuple[tuple[float, float, float, float], ...]
    n_inliers: int


def _partition_flow_tracks(
    prev_pts: np.ndarray,
    next_pts: np.ndarray,
    status: np.ndarray,
    *,
    mad_k: float,
    min_floor_px: float,
    min_valid_points: int,
) -> _FlowPartition:
    good = status.ravel() == 1
    bad = ~good
    prev = prev_pts.reshape(-1, 2)
    nxt = next_pts.reshape(-1, 2)

    lk_rejected = _points_as_dots(prev[bad].reshape(-1, 1, 2)) if bad.any() else ()

    if not good.any():
        return _FlowPartition(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            (),
            lk_rejected,
            0,
        )

    prev_good = prev[good]
    next_good = nxt[good]
    inlier_mask = _displacement_inlier_mask(
        prev_good,
        next_good,
        mad_k=mad_k,
        min_floor_px=min_floor_px,
        min_keep=max(3, min_valid_points),
    )

    prev_in = prev_good[inlier_mask]
    next_in = next_good[inlier_mask]
    prev_out = prev_good[~inlier_mask]
    next_out = next_good[~inlier_mask]

    used = _tracks_from_points(prev_in, next_in)
    outlier_rejected = _tracks_from_points(prev_out, next_out)
    rejected = lk_rejected + outlier_rejected

    return _FlowPartition(
        prev_in,
        next_in,
        used,
        rejected,
        int(len(prev_in)),
    )


def _displacement_inlier_mask(
    prev: np.ndarray,
    nxt: np.ndarray,
    *,
    mad_k: float,
    min_floor_px: float,
    min_keep: int,
) -> np.ndarray:
    disp = nxt.astype(np.float64) - prev.astype(np.float64)
    med_dx = float(np.median(disp[:, 0]))
    med_dy = float(np.median(disp[:, 1]))
    dev = np.hypot(disp[:, 0] - med_dx, disp[:, 1] - med_dy)
    med_dev = float(np.median(dev))
    mad = float(np.median(np.abs(dev - med_dev))) + 1e-6
    thresh = max(float(min_floor_px), float(mad_k) * mad)
    inlier = dev <= thresh
    if int(np.count_nonzero(inlier)) < min_keep and len(dev) >= min_keep:
        order = np.argsort(dev)
        inlier = np.zeros(len(dev), dtype=bool)
        inlier[order[:min_keep]] = True
    return inlier


def _roi_from_bbox(
    shape: tuple[int, ...],
    center_xy: tuple[float, float],
    bbox_wh: tuple[float, float],
) -> tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    bw = max(8, int(round(float(bbox_wh[0]))))
    bh = max(8, int(round(float(bbox_wh[1]))))
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x1 = max(0, cx - bw // 2)
    y1 = max(0, cy - bh // 2)
    x2 = min(w, x1 + bw)
    y2 = min(h, y1 + bh)
    x1 = max(0, x2 - bw)
    y1 = max(0, y2 - bh)
    return x1, y1, x2, y2


def _tracks_from_points(
    prev: np.ndarray,
    nxt: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    if len(prev) == 0:
        return ()
    return tuple(
        (float(prev[i, 0]), float(prev[i, 1]), float(nxt[i, 0]), float(nxt[i, 1]))
        for i in range(len(prev))
    )


def _points_as_dots(pts: np.ndarray | None) -> tuple[tuple[float, float, float, float], ...]:
    if pts is None or len(pts) == 0:
        return ()
    flat = pts.reshape(-1, 2)
    return tuple(
        (float(x), float(y), float(x), float(y)) for x, y in flat
    )
