"""PenTipTrackLocalizer — Localizer Strategy: siamese NN V2/V3 (PyTorch, local weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tracker.types import PenTipTrackOutput
from tracker.config import TrackingConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).parent.parent / "nn" / "configs"
WEIGHTS_DIR = ROOT / "models" / "pretrained"

PENTIPTRACK_VERSIONS: dict[str, tuple[str, str]] = {
    "v2": ("configv2.yaml", "pentiptrackv2.pth"),
    "v3": ("configv3.yaml", "pentiptrackv3.pth"),
}


def _weights_path(filename: str) -> Path:
    dest = WEIGHTS_DIR / filename
    if dest.exists() and dest.stat().st_size > 500_000:
        return dest
    raise RuntimeError(
        f"Missing weights: {dest}. Place {filename} in models/pretrained/."
    )


class PenTipTrackLocalizer:
    """Implements Localizer; EMA template refresh when conf high (maybe_update_neural_template)."""

    _UPDATE_CONF_THRESHOLD = 0.55
    _UPDATE_ALPHA = 0.12

    def __init__(self, version: str = "v2", *, min_peak_score: float | None = None) -> None:
        self._version = version.lower()
        if self._version not in PENTIPTRACK_VERSIONS:
            raise ValueError(f"Unknown PenTipTrack version {version!r}; use v2 or v3")

        import torch

        from tracker.nn.builder import PenTipTrackModel
        from tracker.nn.config import cfg as default_cfg
        from tracker.nn.engine import PenTipTrackEngine

        config_name, weights_name = PENTIPTRACK_VERSIONS[self._version]
        self._cfg = default_cfg.clone()
        self._cfg.merge_from_file(str(CONFIGS / config_name))
        self._cfg.CUDA = bool(torch.cuda.is_available())

        model = PenTipTrackModel(variant=self._version, cfg=self._cfg)
        model = self._load_weights(model, _weights_path(weights_name))
        model.eval()
        if self._cfg.CUDA:
            model = model.cuda()

        self._tracker = PenTipTrackEngine(model, self._cfg)
        self._last_score = 1.0
        self._init_score = 1.0
        self._min_peak_score = float(
            TrackingConfig().peak_lost_threshold
            if min_peak_score is None
            else min_peak_score
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def last_peak_score(self) -> float:
        return float(self._last_score)

    def sync_search_center(self, cx: float, cy: float) -> None:
        self._tracker.center_pos = np.array([float(cx), float(cy)], dtype=np.float64)

    def search_window_roi(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        return self._tracker.search_window_xyxy(frame_shape)

    def initialize(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        w = max(4, int(x2 - x1))
        h = max(4, int(y2 - y1))
        bbox = (float(x1), float(y1), float(w), float(h))
        self._tracker.init(frame, bbox)
        out = self._tracker.track(frame)
        self._last_score = float(out["best_score"])
        self._init_score = max(self._last_score, 0.05)

    def track(self, frame: np.ndarray) -> PenTipTrackOutput | None:
        out = self._tracker.track(frame)
        bbox = out["bbox"]
        score = float(out["best_score"])
        self._last_score = score

        if score < self._min_peak_score:
            return None

        x, y, w, h = bbox
        cx = float(x) + float(w) * 0.5
        cy = float(y) + float(h) * 0.5
        conf = float(np.clip(score / max(self._init_score, 1e-6), 0.0, 1.0))
        return PenTipTrackOutput(
            x=cx,
            y=cy,
            confidence=conf,
            peak_score=score,
            width=float(w),
            height=float(h),
        )

    def maybe_update_neural_template(self, frame: np.ndarray, conf: float) -> None:
        if conf >= self._UPDATE_CONF_THRESHOLD:
            self._refresh_neural_template(frame)

    def _refresh_neural_template(self, frame: np.ndarray) -> None:
        import torch

        t = self._tracker
        cfg = self._cfg
        alpha = float(self._UPDATE_ALPHA)
        w_z = t.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(t.size)
        h_z = t.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(t.size)
        s_z = round(float(np.sqrt(w_z * h_z)))
        z_crop = t._get_subwindow(
            frame,
            t.center_pos,
            cfg.TRACK.EXEMPLAR_SIZE,
            s_z,
            t.channel_average,
        )
        with torch.no_grad():
            new_zf = t.model.backbone(z_crop)
            if t.model.zf is not None:
                t.model.zf = (1.0 - alpha) * t.model.zf + alpha * new_zf
            else:
                t.model.zf = new_zf

    @staticmethod
    def _load_weights(model, weights_path: Path):
        import torch

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for key, val in state.items():
            k = key.split("module.", 1)[-1] if key.startswith("module.") else key
            cleaned[k] = val
        model.load_state_dict(cleaned, strict=False)
        return model
