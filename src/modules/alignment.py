"""
BEV feature alignment modules (AlignNet proxy).
Implements the H0 dense displacement field and the camera→LiDAR
feature projection used in the hypothesis test loss.

Mirrors CRFusion's AlignNet design (Guan et al., 2026, Eq. 7–8):
    Δ = Conv3×3(σ(BN(Conv1×1(Z ⊕ C′))))
with an added Tanh Offset Constraint to bound displacements to the
[-1, 1] normalised grid coordinate system used by F.grid_sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisplacementHead(nn.Module):
    """
    H0: non-rigid per-pixel alignment network (AlignNet proxy).

    Takes the concatenation of camera BEV features (C′) and LiDAR BEV
    features (Z) and predicts a 2-channel dense displacement field Δ.
    The Tanh at the output acts as the Offset Constraint from Fig. 4 of
    CRFusion, bounding Δ to [-1, 1] normalised units.

    Parameters
    ----------
    camera_channels : channels in cam_bev  (256 from FastBEV)
    lidar_channels  : channels in lidar_bev (64 from PointPillarsEncoder)
    """

    def __init__(self, camera_channels: int = 256, lidar_channels: int = 64):
        super().__init__()
        mid_channels = 128
        self.net = nn.Sequential(
            nn.Conv2d(camera_channels + lidar_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1),
            nn.Tanh(),   # Offset Constraint: bounds output to [-1, 1]
        )

    def forward(self, cam_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cam_bev   : [B, camera_channels, H, W]
        lidar_bev : [B, lidar_channels,  H, W]

        Returns
        -------
        delta : [B, 2, H, W]  normalised displacement field in [-1, 1]
        """
        return self.net(torch.cat([cam_bev, lidar_bev], dim=1))


class LidarProjector(nn.Module):
    """
    Learned projection: camera BEV features → LiDAR feature space.

    Maps warped camera BEV features (256ch) into the same channel space
    as the PointPillars LiDAR BEV (64ch) so the alignment loss is computed
    between two structurally equivalent representations rather than a naive
    channel mean vs. binary occupancy map.

    Trained jointly with DisplacementHead each sample.
    Recommended loss: F.mse_loss(projector(warped_cam), lidar_bev)

    Parameters
    ----------
    camera_channels : channels in cam_bev  (256 from FastBEV)
    lidar_channels  : channels in lidar_bev (64 from PointPillarsEncoder)
    """

    def __init__(self, camera_channels: int = 256, lidar_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(camera_channels, lidar_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(lidar_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(lidar_channels, lidar_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(lidar_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, camera_channels, H, W]  warped camera BEV features

        Returns
        -------
        [B, lidar_channels, H, W]
        """
        return self.proj(x)


def apply_dense_warp(features: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Differentiably warp `features` by a normalised displacement field `delta`.

    Parameters
    ----------
    features : [B, C, H, W]
    delta    : [B, 2, H, W]  normalised offsets in [-1, 1] (output of
               DisplacementHead with Tanh).  Channel 0 = x offset,
               channel 1 = y offset.

    Returns
    -------
    warped : [B, C, H, W]

    Notes
    -----
    Constructs an identity sampling grid in [-1, 1] and adds delta
    to produce the final sampling coordinates, then delegates to
    F.grid_sample with bilinear interpolation.
    """
    B, C, H, W = features.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=features.device),
        torch.linspace(-1, 1, W, device=features.device),
        indexing='ij',
    )
    identity_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # delta is already in normalised [-1, 1] units (Tanh output)
    offset = delta.permute(0, 2, 3, 1)   # [B, H, W, 2]
    warped_grid = identity_grid + offset  # [B, H, W, 2]

    return F.grid_sample(features, warped_grid, align_corners=False, mode='bilinear', padding_mode='zeros')
