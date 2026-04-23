"""
CenterPoint detection loss for FastBEV4D.
Focal loss on Gaussian heatmap + L1 on regression outputs at positive centres.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOXEL_SIZE = 0.8    # metres per BEV pixel
GRID_RANGE = 51.2   # BEV extent ±51.2 m
GRID_SIZE  = 128    # pixels (= 2 * GRID_RANGE / VOXEL_SIZE)


def _metre_to_pixel(x_m: float, y_m: float):
    """Convert ego-frame metres to BEV pixel coordinates."""
    px = (x_m + GRID_RANGE) / VOXEL_SIZE
    py = (y_m + GRID_RANGE) / VOXEL_SIZE
    return px, py


def _render_heatmap(gt_boxes, num_classes: int, H: int, W: int,
                    device: torch.device) -> torch.Tensor:
    """
    Render per-class Gaussian heatmaps for a single sample.

    Returns [num_classes, H, W] with values in [0, 1].
    """
    heatmap = torch.zeros(num_classes, H, W, device=device)

    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W] each

    for box in gt_boxes:
        cls = box['class']
        px, py = _metre_to_pixel(box['x'], box['y'])

        if not (0 <= px < W and 0 <= py < H):
            continue

        # Gaussian radius ~ half the smaller box dimension
        radius = max(2.0, min(box['w'], box['l']) / (2.0 * VOXEL_SIZE))
        sigma  = radius / 3.0

        gaussian = torch.exp(
            -((grid_x - px) ** 2 + (grid_y - py) ** 2) / (2 * sigma ** 2)
        )
        heatmap[cls] = torch.maximum(heatmap[cls], gaussian)

    return heatmap


def _focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                alpha: int = 2, beta: int = 4) -> torch.Tensor:
    """
    Modified focal loss from CenterNet (Zhou et al.).
    pred_logits: raw (pre-sigmoid) predictions [num_classes, H, W]
    target:      rendered Gaussian heatmap [num_classes, H, W] in [0, 1]
    """
    pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)

    pos_mask = target.eq(1).float()
    neg_mask = 1.0 - pos_mask
    neg_weight = (1.0 - target).pow(beta)

    pos_loss = -(1 - pred).pow(alpha) * pred.log() * pos_mask
    neg_loss = -neg_weight * pred.pow(alpha) * (1 - pred).log() * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class CenterPointLoss(nn.Module):
    """
    CenterPoint loss for FastBEV4D.

    Combines:
      - Focal loss on the class heatmap
      - L1 on (reg, height, dim, rot) at GT object centres
    """

    def __init__(self, num_classes: int = 10,
                 weight_heatmap: float = 1.0,
                 weight_reg: float = 0.25):
        super().__init__()
        self.num_classes     = num_classes
        self.weight_heatmap  = weight_heatmap
        self.weight_reg      = weight_reg

    def forward(self, predictions, gt_boxes_batch):
        """
        Args:
            predictions:    list of task dicts from CenterHead.forward()
            gt_boxes_batch: list[list[dict]] — one list per batch item
        Returns:
            dict with 'loss', 'heatmap_loss', 'reg_loss'
        """
        task   = predictions[0]
        B      = task['heatmap'].shape[0]
        H, W   = task['heatmap'].shape[2], task['heatmap'].shape[3]
        device = task['heatmap'].device

        hm_loss  = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            gt = gt_boxes_batch[b]

            # Heatmap
            target_hm = _render_heatmap(gt, self.num_classes, H, W, device)
            hm_loss  += _focal_loss(task['heatmap'][b], target_hm)

            # Regression at GT centres
            if gt:
                reg_loss += self._regression_loss(task, b, gt, H, W, device)

        hm_loss  /= B
        reg_loss /= B
        total     = self.weight_heatmap * hm_loss + self.weight_reg * reg_loss

        return {'loss': total, 'heatmap_loss': hm_loss, 'reg_loss': reg_loss}

    @staticmethod
    def _regression_loss(task, b, gt_boxes, H, W, device):
        loss = torch.tensor(0.0, device=device)
        n    = 0

        for box in gt_boxes:
            px, py = _metre_to_pixel(box['x'], box['y'])
            xi, yi = int(px), int(py)

            if not (0 <= xi < W and 0 <= yi < H):
                continue

            # Sub-pixel centre offset
            gt_reg = torch.tensor([px - xi, py - yi], device=device)
            loss  += F.l1_loss(task['reg'][b, :, yi, xi], gt_reg)

            # Height (z)
            loss  += F.l1_loss(
                task['height'][b, 0, yi, xi],
                torch.tensor(box['z'], device=device),
            )

            # Log-dimensions
            gt_dim = torch.tensor(
                [np.log(max(box['w'], 1e-4)),
                 np.log(max(box['l'], 1e-4)),
                 np.log(max(box['h'], 1e-4))],
                device=device,
            )
            loss += F.l1_loss(task['dim'][b, :, yi, xi], gt_dim)

            # Rotation (sin / cos encoding)
            gt_rot = torch.tensor(
                [np.sin(box['yaw']), np.cos(box['yaw'])], device=device
            )
            loss += F.l1_loss(task['rot'][b, :, yi, xi], gt_rot)

            # Velocity in ego frame (NaN instances zeroed in load_gt_boxes)
            gt_vel = torch.tensor([box['vx'], box['vy']], device=device)
            loss  += F.l1_loss(task['vel'][b, :, yi, xi], gt_vel)

            n += 1

        return loss / max(n, 1)
