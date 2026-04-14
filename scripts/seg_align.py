"""
seg_align.py — End-to-end AlignNet + SegHead for training with segmentation supervision.

Replaces the per-sample MSE optimization with a proper end-to-end training
pipeline where the only supervision signal is downstream segmentation
performance.

Architecture
------------
AlignNet  : concat(cam_bev, lidar_bev) → Conv1x1-BN-ReLU → Conv3x3 → Tanh
            Predicts per-pixel offset Δ in normalised [-1,1] units.
            cam_bev is warped by Δ to produce cam_bev_aligned.

SegHead   : concat(cam_bev_aligned, lidar_bev) → 3x Conv-BN-ReLU → Conv1x1
            Outputs [B, NUM_CLASSES, H, W] logits.
            Loss: binary cross-entropy per channel (multi-label).

Training signal
---------------
The segmentation loss propagates gradients back through the warp (via
F.grid_sample which is differentiable) into AlignNet. AlignNet therefore
learns to predict offsets that improve segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 3   # lane_divider, ped_crossing, road_segment


class AlignNet(nn.Module):
    """
    Lightweight offset predictor that mirrors CRFusion Eq. 7 (Guan et al., 2026).

    Parameters
    ----------
    camera_channels : channels in cam_bev  (256 from FastBEV)
    lidar_channels  : channels in lidar_bev (256 from PointPillars)
    mid_channels    : internal bottleneck width
    max_offset      : Tanh output is scaled by this to limit warp magnitude
    """

    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels:  int = 256,
        mid_channels:    int = 128,
        max_offset:      float = 0.1,   # ~6.4m at 0.8m/px over ±51.2m range
    ):
        super().__init__()
        self.max_offset = max_offset

        self.net = nn.Sequential(
            # 1×1 conv — channel compression + cross-modal mixing
            nn.Conv2d(camera_channels + lidar_channels, mid_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3×3 conv — spatial context for offset prediction
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # output: 2-channel offset field
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1),
            nn.Tanh(),   # bounds output to [-1, 1], scaled by max_offset
        )

        # Initialise output conv to near-zero so training starts from
        # identity warp (no alignment) rather than random offsets
        nn.init.xavier_uniform_(self.net[-2].weight, gain=0.01)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, cam_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cam_bev   : [B, camera_channels, H, W]
        lidar_bev : [B, lidar_channels,  H, W]

        Returns
        -------
        delta : [B, 2, H, W]  normalised offset in [-max_offset, max_offset]
        """
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        return self.net(x) * self.max_offset


class SegHead(nn.Module):
    """
    Simple BEV segmentation head.

    Takes the concatenation of aligned camera BEV and LiDAR BEV features
    and produces per-pixel class logits.

    Parameters
    ----------
    camera_channels : channels in cam_bev_aligned (256)
    lidar_channels  : channels in lidar_bev (256)
    num_classes     : number of output classes (3)
    mid_channels    : internal feature width
    """

    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels:  int = 256,
        num_classes:     int = NUM_CLASSES,
        mid_channels:    int = 128,
    ):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.head = nn.Sequential(
            _block(camera_channels + lidar_channels, mid_channels),
            _block(mid_channels, mid_channels // 2),
            nn.Conv2d(mid_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, cam_bev_aligned: torch.Tensor,
                lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cam_bev_aligned : [B, camera_channels, H, W]
        lidar_bev       : [B, lidar_channels,  H, W]

        Returns
        -------
        logits : [B, num_classes, H, W]  (raw, before sigmoid)
        """
        x = torch.cat([cam_bev_aligned, lidar_bev], dim=1)
        return self.head(x)


class BEVAlignSegNet(nn.Module):
    """
    Full end-to-end module: AlignNet + warp + SegHead.

    This is the trainable component. FastBEV and PointPillars are frozen
    upstream — this module only contains AlignNet and SegHead parameters.

    Usage
    -----
    model = BEVAlignSegNet()
    logits, delta = model(cam_bev, lidar_bev)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()   # gradients flow through SegHead → warp → AlignNet
    """

    def __init__(
        self,
        camera_channels: int = 256,
        lidar_channels:  int = 256,
        num_classes:     int = NUM_CLASSES,
        mid_channels:    int = 128,
        max_offset:      float = 0.1,
        no_align:        bool = False   # ablation: disable alignment by zeroing offsets
    ):
        super().__init__()
        self.no_align = no_align
        self.align = AlignNet(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            mid_channels=mid_channels,
            max_offset=max_offset,
        )
        self.seg = SegHead(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            num_classes=num_classes,
            mid_channels=mid_channels,
        )

    def _warp(self, features: torch.Tensor,
              delta: torch.Tensor) -> torch.Tensor:
        """Differentiable warp of `features` by normalised offset `delta`."""
        B, C, H, W = features.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij',
        )
        identity = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        offset   = delta.permute(0, 2, 3, 1)   # [B, H, W, 2]
        grid     = identity + offset
        return F.grid_sample(features, grid, align_corners=False,
                             mode='bilinear', padding_mode='zeros')

    def forward(self, cam_bev: torch.Tensor,
                lidar_bev: torch.Tensor):
        """
        Parameters
        ----------
        cam_bev   : [B, camera_channels, H, W]  frozen FastBEV features
        lidar_bev : [B, lidar_channels,  H, W]  frozen PointPillars features

        Returns
        -------
        logits : [B, num_classes, H, W]  segmentation logits
        delta  : [B, 2, H, W]           predicted offset field
        """
        if self.no_align:
            delta = torch.zeros(
                cam_bev.shape[0], 2, cam_bev.shape[2], cam_bev.shape[3],
                device=cam_bev.device
            )
            logits = self.seg(cam_bev, lidar_bev)
        else:
            delta           = self.align(cam_bev, lidar_bev)
            cam_bev_aligned = self._warp(cam_bev, delta)
            logits          = self.seg(cam_bev_aligned, lidar_bev)
        return logits, delta


def seg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for multi-label BEV segmentation.

    Each channel is treated as an independent binary classification —
    a pixel can simultaneously be a lane divider AND a road segment.

    Parameters
    ----------
    logits     : [B, C, H, W]  raw logits (before sigmoid)
    labels     : [B, C, H, W]  float {0.0, 1.0} ground truth
    pos_weight : [C]           per-class weight for class imbalance
                               (road_segment is dense, ped_crossing sparse)

    Returns
    -------
    scalar loss
    """
    if pos_weight is not None:
        pos_weight = pos_weight.view(-1, 1, 1)
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction='mean'
    )


def compute_pos_weights(labels_list, num_classes: int = NUM_CLASSES,
                        device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Compute per-class positive weights from a list of label tensors.
    Used to down-weight dense classes (road_segment) relative to sparse
    ones (ped_crossing).

    pos_weight[c] = (num_negative_pixels[c]) / (num_positive_pixels[c])

    Parameters
    ----------
    labels_list : list of [C, H, W] tensors
    num_classes : number of classes

    Returns
    -------
    [C] float tensor of positive weights, clamped to [1, 20]
    """
    pos = torch.zeros(num_classes)
    neg = torch.zeros(num_classes)
    for lbl in labels_list:
        pos += lbl.sum(dim=(-1, -2)).cpu()
        neg += (1 - lbl).sum(dim=(-1, -2)).cpu()
    weights = (neg / (pos + 1e-6)).clamp(1.0, 20.0)
    return weights.to(device)