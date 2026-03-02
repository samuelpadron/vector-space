"""
PointPillars-style LiDAR → BEV encoder.
Pure PyTorch re-implementation to deprecated mmdet3d PointPillars.
Matches the preprocessing pipeline described in CRFusion (Guan et al., 2026).

Pipeline
--------
1. Pillarisation       — assign each point to an (xi, yi) BEV cell.
2. Point augmentation  — append per-point offsets (Δx, Δy, Δz) from
                         pillar centre. Feature dim = 7: x, y, z,
                         intensity, Δx, Δy, Δz.
3. PointNet MLP        — shared Linear→BN→ReLU lifts each point to
                         `pillar_channels` dims.
4. Max-pool            — one descriptor per non-empty pillar.
5. Scatter             — place descriptors back onto the 2-D BEV grid,
                         producing a pseudo-image [1, C_pillar, H, W].
6. BEV backbone        — two stride-1 Conv-BN-ReLU blocks refine the
                         pseudo-image → [1, out_channels, H, W].
                         Stride-1 keeps H×W = grid resolution so spatial
                         dims always match cam_bev without interpolation.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes


class HandcraftedLidarBEV:
    """
    Deterministic 4-channel LiDAR BEV encoder.

    Not an nn.Module — contains no learnable parameters.
    Call encode() to produce a BEV tensor from a raw point cloud.

    Parameters
    ----------
    grid_conf : dict with keys 'xbound', 'ybound', each [min, max, res].
                Should be identical to the camera BEV grid so spatial
                dimensions match without interpolation.

    Example
    -------
    >>> encoder = HandcraftedLidarBEV(
    ...     grid_conf={'xbound': [-51.2, 51.2, 0.8],
    ...                'ybound': [-51.2, 51.2, 0.8]}
    ... )
    >>> points = load_lidar_points(nusc, token)          # (N, 4)
    >>> lidar_bev = encoder.encode(points, device)       # (1, 4, 128, 128)
    """

    NUM_CHANNELS = 4   # occupancy, max_height, density, mean_intensity

    def __init__(self, grid_conf: dict):
        x_min, x_max, x_res = grid_conf['xbound']
        y_min, y_max, y_res = grid_conf['ybound']

        self.x_min, self.x_max, self.x_res = x_min, x_max, x_res
        self.y_min, self.y_max, self.y_res = y_min, y_max, y_res
        self.nx = int(round((x_max - x_min) / x_res))   # 128
        self.ny = int(round((y_max - y_min) / y_res))   # 128

    @property
    def out_channels(self) -> int:
        return self.NUM_CHANNELS

    def encode(self, points: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Build a 4-channel BEV descriptor from a raw point cloud.

        Parameters
        ----------
        points : (N, 4) float32 array [x, y, z, intensity] in ego frame.
        device : torch device for the output tensor.

        Returns
        -------
        Tensor [1, 4, ny, nx] on `device`, values in [0, 1].
        """
        bev = np.zeros((self.NUM_CHANNELS, self.ny, self.nx), dtype=np.float32)

        # Range filter
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max)
        )
        pts = points[mask]   # (M, 4)

        if len(pts) == 0:
            return torch.from_numpy(bev).unsqueeze(0).to(device)

        # Compute cell indices
        xi = np.clip(
            np.floor((pts[:, 0] - self.x_min) / self.x_res).astype(np.int32),
            0, self.nx - 1,
        )
        yi = np.clip(
            np.floor((pts[:, 1] - self.y_min) / self.y_res).astype(np.int32),
            0, self.ny - 1,
        )

        z         = pts[:, 2]
        intensity = pts[:, 3]

        # Occupancy 
        bev[0, yi, xi] = 1.0

        # Max height (normalised)
        z_min, z_max = -3.0, 5.0   # ego-frame z range
        z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
        np.maximum.at(bev[1], (yi, xi), z_norm)

        # Point density
        count_map = np.zeros((self.ny, self.nx), dtype=np.float32)
        np.add.at(count_map, (yi, xi), 1.0)
        max_count = count_map.max()
        if max_count > 0:
            bev[2] = count_map / max_count

        # Mean intensity 
        intensity_sum = np.zeros((self.ny, self.nx), dtype=np.float32)
        np.add.at(intensity_sum, (yi, xi), intensity)
        # Divide sum by count (avoid div-by-zero on empty cells)
        nonzero = count_map > 0
        bev[3, nonzero] = intensity_sum[nonzero] / count_map[nonzero]
        # nuScenes intensity is in [0, 255] — normalise to [0, 1]
        bev[3] = np.clip(bev[3] / 255.0, 0.0, 1.0)

        return torch.from_numpy(bev).unsqueeze(0).to(device)   # (1, 4, ny, nx)
    
    
class PointPillarsEncoder(nn.Module):
    """
    Parameters
    ----------
    grid_conf             : dict with keys 'xbound', 'ybound',
                            each [min, max, resolution].
    pillar_channels       : intermediate pillar feature dimension.
    out_channels          : output BEV channels — set equal to
                            cam_bev channels so downstream modules
                            (DisplacementHead, LidarProjector) can
                            concatenate/compare directly.
    max_points_per_pillar : maximum points sampled per pillar;
                            excess points are discarded.
    """

    POINT_DIM = 7  # x, y, z, intensity, Δx, Δy, Δz

    def __init__(
        self,
        grid_conf: dict,
        pillar_channels: int = 64,
        out_channels: int = 64,
        max_points_per_pillar: int = 32,
    ):
        super().__init__()
        x_min, x_max, x_res = grid_conf['xbound']
        y_min, y_max, y_res = grid_conf['ybound']

        self.x_min, self.x_max, self.x_res = x_min, x_max, x_res
        self.y_min, self.y_max, self.y_res = y_min, y_max, y_res
        self.nx = int((x_max - x_min) / x_res)   # BEV width  (128 for nuScenes default)
        self.ny = int((y_max - y_min) / y_res)   # BEV height (128 for nuScenes default)
        self.max_pts = max_points_per_pillar
        self.pillar_channels = pillar_channels
        self.out_channels = out_channels

        self.pointnet = nn.Sequential(
            nn.Linear(self.POINT_DIM, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, pillar_channels, bias=False),
            nn.BatchNorm1d(pillar_channels),
            nn.ReLU(inplace=True),
        )

        # BEV backbone 
        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.bev_backbone = nn.Sequential(
            _block(pillar_channels, pillar_channels * 2),
            _block(pillar_channels * 2, out_channels),
        )
        

    def _pillarise(self, points: np.ndarray):
        """
        Convert raw (N, 4) point cloud [x, y, z, intensity] to
        pillar feature tensors.

        Returns
        -------
        pillar_feats  : FloatTensor [P, max_pts, POINT_DIM]  (zero-padded)
        pillar_coords : LongTensor  [P, 2]                   (xi, yi)
        """
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max)
        )
        pts = points[mask]

        if len(pts) == 0:
            return (
                torch.zeros(0, self.max_pts, self.POINT_DIM),
                torch.zeros(0, 2, dtype=torch.long),
            )

        xi = np.clip(np.floor((pts[:, 0] - self.x_min) / self.x_res).astype(np.int32), 0, self.nx - 1)
        yi = np.clip(np.floor((pts[:, 1] - self.y_min) / self.y_res).astype(np.int32), 0, self.ny - 1)

        flat_idx = yi * self.nx + xi
        cell_ids, inverse = np.unique(flat_idx, return_inverse=True)
        num_pillars = len(cell_ids)

        cell_xi = (cell_ids % self.nx).astype(np.float32)
        cell_yi = (cell_ids // self.nx).astype(np.float32)
        cx = cell_xi * self.x_res + self.x_min + self.x_res / 2.0
        cy = cell_yi * self.y_res + self.y_min + self.y_res / 2.0

        pillar_feats = np.zeros((num_pillars, self.max_pts, self.POINT_DIM), dtype=np.float32)
        counts = np.zeros(num_pillars, dtype=np.int32)

        for pt_i, pillar_i in enumerate(inverse):
            k = counts[pillar_i]
            if k >= self.max_pts:
                continue
            x, y, z, intensity = pts[pt_i]
            pillar_feats[pillar_i, k] = [x, y, z, intensity,
                                         x - cx[pillar_i],
                                         y - cy[pillar_i],
                                         z]  # Δz from ground (flat assumption)
            counts[pillar_i] += 1

        pillar_coords = np.stack(
            [(cell_ids % self.nx).astype(np.int64),
             (cell_ids // self.nx).astype(np.int64)],
            axis=1,
        )  # (P, 2) — (xi, yi)

        return torch.from_numpy(pillar_feats), torch.from_numpy(pillar_coords)

    def _encode_pillars(self, pillar_feats: torch.Tensor) -> torch.Tensor:
        """
        PointNet MLP + max-pool over points inside each pillar.

        pillar_feats : (P, max_pts, POINT_DIM)
        returns      : (P, pillar_channels)
        """
        P, N, D = pillar_feats.shape
        flat = pillar_feats.view(P * N, D)
        encoded = self.pointnet(flat).view(P, N, self.pillar_channels)
        return encoded.max(dim=1).values   # (P, C)

    def _scatter_to_bev(
        self,
        descriptors: torch.Tensor,
        pillar_coords: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Scatter (P, C) pillar descriptors onto a [1, C, ny, nx] BEV grid.
        Non-empty pillars are written; empty cells remain zero.
        """
        C = descriptors.shape[1]
        bev = torch.zeros(1, C, self.ny, self.nx, device=device, dtype=descriptors.dtype)
        if len(pillar_coords) > 0:
            xi = pillar_coords[:, 0]
            yi = pillar_coords[:, 1]
            bev[0, :, yi, xi] = descriptors.T   # (C, P) broadcast
        return bev


    def forward(self, points: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Parameters
        ----------
        points : (N, 4) numpy float32 array [x, y, z, intensity] in ego frame
        device : torch device to place output on

        Returns
        -------
        Tensor [1, out_channels, ny, nx] — rich multi-channel LiDAR BEV
        """
        pillar_feats, pillar_coords = self._pillarise(points)

        if len(pillar_coords) == 0:
            return torch.zeros(1, self.out_channels, self.ny, self.nx, device=device)

        pillar_feats = pillar_feats.to(device)
        pillar_coords = pillar_coords.to(device)

        descriptors = self._encode_pillars(pillar_feats)        # (P, C_pillar)
        pseudo_img = self._scatter_to_bev(descriptors, pillar_coords, device)  # (1, C_pillar, ny, nx)
        return self.bev_backbone(pseudo_img)                    # (1, out_channels, ny, nx)


def load_lidar_points(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load the raw LiDAR point cloud for a nuScenes sample.

    nuScenes .pcd.bin files store 5 floats per point; the 5th is the
    ring/beam index which is dropped here.

    Returns
    -------
    (N, 4) float32 array: [x, y, z, intensity] in ego/LiDAR frame.
    """
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = Path(nusc.dataroot) / lidar_data['filename']
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
