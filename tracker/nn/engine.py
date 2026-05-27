from __future__ import annotations

import cv2
import numpy as np
import torch

from tracker.nn.coords import corner2center


class PenTipTrackEngine:
    """Siamese PenTipTrack per-frame tracker (exemplar + search)."""

    def __init__(self, model, cfg) -> None:
        self.cfg = cfg
        self.score_size = int(cfg.TRACK.OUTPUT_SIZE)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()
        self.points = self._generate_points(self.cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()
        self.center_pos = np.zeros(2, dtype=np.float64)
        self.size = np.zeros(2, dtype=np.float64)
        self.channel_average = np.zeros(3, dtype=np.float64)

    def _generate_points(self, stride: int, size: int) -> np.ndarray:
        ori = -(size // 2) * stride
        x, y = np.meshgrid(
            [ori + stride * dx for dx in np.arange(0, size)],
            [ori + stride * dy for dy in np.arange(0, size)],
        )
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        return points

    def init(self, img: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        x, y, w, h = bbox
        self.center_pos = np.array([x + (w - 1) / 2.0, y + (h - 1) / 2.0])
        self.size = np.array([w, h], dtype=np.float64)
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(float(np.sqrt(w_z * h_z)))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self._get_subwindow(img, self.center_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img: np.ndarray) -> dict:
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self._get_subwindow(
            img,
            self.center_pos,
            self.cfg.TRACK.INSTANCE_SIZE,
            round(float(s_x)),
            self.channel_average,
        )
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self.points)

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        r_c = change((self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        best_idx = int(np.argmax(pscore))
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        out_bbox = [cx - width / 2, cy - height / 2, width, height]
        return {"bbox": out_bbox, "best_score": float(score[best_idx])}

    def search_window_xyxy(self, img_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Image-space bounds of the PenTipTrack search crop (before resize to INSTANCE_SIZE)."""
        h_img, w_img = int(img_shape[0]), int(img_shape[1])
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = float(np.sqrt(w_z * h_z))
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        sz = float(round(s_x))
        c = (sz + 1.0) / 2.0
        pos = self.center_pos
        x1 = int(np.floor(float(pos[0]) - c + 0.5))
        y1 = int(np.floor(float(pos[1]) - c + 0.5))
        x2 = int(np.floor(x1 + sz - 1.0))
        y2 = int(np.floor(y1 + sz - 1.0))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2 + 1)
        y2 = min(h_img, y2 + 1)
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)

    def _get_subwindow(
        self,
        im: np.ndarray,
        pos: np.ndarray,
        model_sz: int,
        original_sz: float,
        avg_chans: np.ndarray,
    ) -> torch.Tensor:
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))
        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        r, c_im, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c_im + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad : top_pad + r, left_pad : left_pad + c_im, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad : left_pad + c_im, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad :, left_pad : left_pad + c_im, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c_im + left_pad :, :] = avg_chans
            im_patch = te_im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        else:
            im_patch = im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        tensor = torch.from_numpy(im_patch)
        if self.cfg.CUDA:
            tensor = tensor.cuda()
        return tensor

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
        return score.softmax(1).detach()[:, 1].cpu().numpy()

    @staticmethod
    def _bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height
