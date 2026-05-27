from __future__ import annotations

import torch.nn as nn

from tracker.nn.backbone import get_backbone
from tracker.nn.head import get_ban_head
from tracker.nn.neck import get_neck


class PenTipTrackModel(nn.Module):
    """Inference-only PenTipTrack siamese model."""

    def __init__(self, *, variant: str = "v2", cfg=None) -> None:
        super().__init__()
        if cfg is None:
            from tracker.nn.config import cfg as default_cfg
            cfg = default_cfg
        self.cfg = cfg
        self.backbone = get_backbone(cfg.BACKBONE.TYPE, **cfg.BACKBONE.KWARGS)
        self.neck = None
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)
        self.ban_head = None
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE, variant=variant, **cfg.BAN.KWARGS)
        self.zf = None

    def template(self, z) -> None:
        self.zf = self.backbone(z)

    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.ban_head(self.zf, xf)
        return {"cls": cls, "loc": loc}
