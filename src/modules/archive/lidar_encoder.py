"""
Hand-crafted deterministic LiDAR BEV encoder.

Replaces the random-weight PointPillarsEncoder with a training-free
4-channel descriptor computed directly from raw point cloud geometry.
This gives the DisplacementHead a stable, geometry-grounded target so
the alignment loss has meaningful gradient signal.

Channels
--------
  0 — Occupancy      : 1.0 if any point falls in the cell, else 0.
  1 — Max height     : highest z value in the cell (normalised to [0,1]).
                       Distinguishes vehicles, walls, ground plane.
  2 — Point density  : number of points in cell / max count across grid,
                       clipped to [0, 1]. Dense = solid surface.
  3 — Mean intensity : average LiDAR return intensity, normalised to
                       [0, 1]. Reflectivity signature of surfaces.

All channels are computed with vectorised NumPy — no PyTorch, no
training, no randomness. Output is a float32 tensor [1, 4, ny, nx]
on the requested device.

Scientific justification
------------------------
The hypothesis test asks whether the geometric misalignment between
camera BEV and LiDAR BEV is well-approximated by a global Sim(2)
transform.  For this test to be valid the LiDAR representation must
reflect actual 3-D geometry, not random projections.  The four chosen
channels encode complementary geometric cues present in every nuScenes
frame and require no learned parameters.
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

        # ── 1. Range filter ───────────────────────────────────────────────
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max)
        )
        pts = points[mask]   # (M, 4)

        if len(pts) == 0:
            return torch.from_numpy(bev).unsqueeze(0).to(device)

        # ── 2. Compute cell indices ───────────────────────────────────────
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

        # ── Ch 0: Occupancy ───────────────────────────────────────────────
        bev[0, yi, xi] = 1.0

        # ── Ch 1: Max height (normalised) ─────────────────────────────────
        # Use np.maximum.at for per-cell max (scatter reduce)
        z_min, z_max = -3.0, 5.0   # nuScenes typical ego-frame z range
        z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
        np.maximum.at(bev[1], (yi, xi), z_norm)

        # ── Ch 2: Point density (normalised) ─────────────────────────────
        count_map = np.zeros((self.ny, self.nx), dtype=np.float32)
        np.add.at(count_map, (yi, xi), 1.0)
        max_count = count_map.max()
        if max_count > 0:
            bev[2] = count_map / max_count

        # ── Ch 3: Mean intensity ──────────────────────────────────────────
        intensity_sum = np.zeros((self.ny, self.nx), dtype=np.float32)
        np.add.at(intensity_sum, (yi, xi), intensity)
        # Divide sum by count (avoid div-by-zero on empty cells)
        nonzero = count_map > 0
        bev[3, nonzero] = intensity_sum[nonzero] / count_map[nonzero]
        # nuScenes intensity is in [0, 255] — normalise to [0, 1]
        bev[3] = np.clip(bev[3] / 255.0, 0.0, 1.0)

        return torch.from_numpy(bev).unsqueeze(0).to(device)   # (1, 4, ny, nx)


def load_lidar_points(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load the raw LiDAR point cloud for a nuScenes sample.

    nuScenes .pcd.bin files store 5 floats per point; the 5th is the
    ring/beam index which is dropped here.

    Returns
    -------
    (N, 4) float32 array: [x, y, z, intensity] in LiDAR/ego frame.
    """
    sample    = nusc.get('sample', sample_token)
    lidar_sd  = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = Path(nusc.dataroot) / lidar_sd['filename']
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]


def _make_conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _VFE(nn.Module):
    """
    Voxel Feature Encoder matching pts_voxel_encoder in the checkpoint.
    Two linear+BN layers: 10→64, 64→64 (with max-pool between).
    Input feature dim is 10: x,y,z,intensity + 6 pillar-relative offsets
    as used in the hv_pointpillars_fpn_sbn nuScenes config.
    """
    def __init__(self):
        super().__init__()
        # Layer 0: 10 → 64
        self.norm0   = nn.BatchNorm1d(64)
        self.linear0 = nn.Linear(10, 64, bias=False)
        # Layer 1: 64 → 64 (after max-pool)
        self.norm1   = nn.BatchNorm1d(64)
        self.linear1 = nn.Linear(64, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [P, N, 10]
        P, N, D = x.shape
        out = self.linear0(x.view(P * N, D))    # [P*N, 64]
        out = self.norm0(out)
        out = torch.relu(out)
        out = out.view(P, N, 64).max(dim=1).values  # [P, 64] max-pool
        out = self.linear1(out)                  # [P, 64]
        out = self.norm1(out)
        out = torch.relu(out)
        return out


class _SECOND_Backbone(nn.Module):
    """
    3-block SECOND backbone matching pts_backbone in the checkpoint.
    Block 0: stride-2, 4 convs, 64→64
    Block 1: stride-2, 5 convs, 64→128
    Block 2: stride-2, 6 convs, 128→256
    Layer pattern per block: Conv,BN,ReLU repeated, first conv is stride-2.
    """
    def __init__(self):
        super().__init__()

        def _block(in_ch, out_ch, n_convs):
            layers = []
            for i in range(n_convs):
                stride = 2 if i == 0 else 1
                cin    = in_ch if i == 0 else out_ch
                layers += [
                    nn.Conv2d(cin, out_ch, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        self.blocks = nn.ModuleList([
            _block(64,  64,  4),
            _block(64,  128, 5),
            _block(128, 256, 6),
        ])

    def forward(self, x: torch.Tensor):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs   # [C0, C1, C2] at 1/2, 1/4, 1/8 of input resolution


class _FPN(nn.Module):
    """
    FPN neck matching pts_neck in the checkpoint.
    3 lateral convs (1×1) + 3 fpn convs (3×3) all outputting 256ch.
    Top-down feature fusion with bilinear upsample.
    """
    def __init__(self):
        super().__init__()

        def _lateral(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        def _fpn_conv():
            return nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        self.lateral_convs = nn.ModuleList([
            _lateral(64),
            _lateral(128),
            _lateral(256),
        ])
        self.fpn_convs = nn.ModuleList([_fpn_conv() for _ in range(3)])

    def forward(self, features: list) -> torch.Tensor:
        # features: [C0@H/2, C1@H/4, C2@H/8]
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Upsample all to largest scale and sum
        target_size = outs[0].shape[-2:]
        fused = outs[0]
        for o in outs[1:]:
            fused = fused + nn.functional.interpolate(o, size=target_size, mode='bilinear', align_corners=False)
        return fused   # [1, 256, H/2, W/2]


class PretrainedPointPillars(nn.Module):
    """
    Full PointPillars encoder loading weights from the MMDetection3D
    hv_pointpillars_fpn_sbn-all nuScenes checkpoint.

    Produces a [1, 256, H, W] LiDAR BEV feature map from raw point cloud.
    The output is bilinearly upsampled to match cam_bev spatial dimensions
    (default 128×128) so no interpolation is needed downstream.

    Parameters
    ----------
    checkpoint_path : path to the .pth checkpoint file
    out_spatial_size: (H, W) to upsample output to — should match cam_bev
    voxel_size      : metres per pillar cell (0.2m matches the checkpoint config)
    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Usage
    -----
    encoder = PretrainedPointPillars(checkpoint_path='models/pointpillars/...pth')
    encoder = encoder.to(device).eval()
    lidar_bev = encoder(points, device)   # [1, 256, 128, 128]
    """

    # nuScenes PointPillars config
    VOXEL_SIZE       = 0.2    # metres
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    POINT_DIM        = 10    # x,y,z,intensity + cx,cy,cz offsets + pillar_x,pillar_y,pillar_z

    def __init__(
        self,
        checkpoint_path: str,
        out_spatial_size: tuple = (128, 128),
        max_points_per_pillar: int = 20,
        max_pillars: int = 30000,
    ):
        super().__init__()
        self.out_spatial_size       = out_spatial_size
        self.max_pts                = max_points_per_pillar
        self.max_pillars            = max_pillars

        x_min, y_min, z_min, x_max, y_max, z_max = self.POINT_CLOUD_RANGE
        vs = self.VOXEL_SIZE
        self.nx = int((x_max - x_min) / vs)   # 512
        self.ny = int((y_max - y_min) / vs)   # 512
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max

        self.vfe      = _VFE()
        self.backbone = _SECOND_Backbone()
        self.neck     = _FPN()

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str):
        ckpt       = torch.load(path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Remap checkpoint keys: remove module prefix, keep only our modules
        own_state = self.state_dict()
        mapping = {
            'pts_voxel_encoder.': 'vfe.',
            'pts_backbone.':      'backbone.',
            'pts_neck.':          'neck.',
        }
        loaded, skipped = 0, 0
        remapped = {}
        for ck, cv in state_dict.items():
            new_key = None
            for prefix, new_prefix in mapping.items():
                if ck.startswith(prefix):
                    new_key = new_prefix + ck[len(prefix):]
                    break
            if new_key is not None and new_key in own_state:
                if own_state[new_key].shape == cv.shape:
                    remapped[new_key] = cv.float()
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        self.load_state_dict(remapped, strict=False)
        print(f"  PretrainedPointPillars: loaded {loaded} keys, skipped {skipped}")

    def _pillarise(self, points: np.ndarray):
        """
        Voxelise points and compute 10-dim per-point features matching
        the nuScenes PointPillars config:
        [x, y, z, intensity, x-cx, y-cy, z-cz, cx, cy, cz]
        where (cx, cy, cz) is the pillar centre.
        """
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] < self.z_max)
        )
        pts = points[mask]
        if len(pts) == 0:
            return (torch.zeros(1, self.max_pts, self.POINT_DIM),
                    torch.zeros(1, 2, dtype=torch.long))

        xi = np.clip(
            np.floor((pts[:, 0] - self.x_min) / self.VOXEL_SIZE).astype(np.int32),
            0, self.nx - 1,
        )
        yi = np.clip(
            np.floor((pts[:, 1] - self.y_min) / self.VOXEL_SIZE).astype(np.int32),
            0, self.ny - 1,
        )

        flat_idx           = yi * self.nx + xi
        cell_ids, inverse  = np.unique(flat_idx, return_inverse=True)

        # Limit number of pillars
        if len(cell_ids) > self.max_pillars:
            cell_ids = cell_ids[:self.max_pillars]
            keep     = inverse < self.max_pillars
            pts, xi, yi, inverse = pts[keep], xi[keep], yi[keep], inverse[keep]

        num_pillars = len(cell_ids)
        cell_xi     = (cell_ids % self.nx).astype(np.float32)
        cell_yi     = (cell_ids // self.nx).astype(np.float32)
        cx          = cell_xi * self.VOXEL_SIZE + self.x_min + self.VOXEL_SIZE / 2
        cy          = cell_yi * self.VOXEL_SIZE + self.y_min + self.VOXEL_SIZE / 2
        cz          = (self.z_min + self.z_max) / 2.0

        feats  = np.zeros((num_pillars, self.max_pts, self.POINT_DIM), dtype=np.float32)
        counts = np.zeros(num_pillars, dtype=np.int32)

        for pt_i, pillar_i in enumerate(inverse):
            if pillar_i >= num_pillars:
                continue
            k = counts[pillar_i]
            if k >= self.max_pts:
                continue
            x, y, z, intensity = pts[pt_i, :4]
            feats[pillar_i, k] = [
                x, y, z, intensity,
                x - cx[pillar_i],
                y - cy[pillar_i],
                z - cz,
                cx[pillar_i], cy[pillar_i], cz,
            ]
            counts[pillar_i] += 1

        coords = np.stack([
            (cell_ids % self.nx).astype(np.int64),
            (cell_ids // self.nx).astype(np.int64),
        ], axis=1)

        return torch.from_numpy(feats), torch.from_numpy(coords)

    def forward(self, points: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Parameters
        ----------
        points : (N, 4) numpy float32 [x, y, z, intensity] in ego frame
        device : target device

        Returns
        -------
        [1, 256, H, W]  LiDAR BEV features upsampled to out_spatial_size
        """
        pillar_feats, pillar_coords = self._pillarise(points)
        pillar_feats  = pillar_feats.to(device)
        pillar_coords = pillar_coords.to(device)

        # VFE: encode each pillar
        P, N, D = pillar_feats.shape
        flat        = pillar_feats.view(P * N, D)
        vfe_out     = self.vfe(pillar_feats)          # [P, 64]

        # Scatter to pseudo-image [1, 64, ny, nx]
        pseudo = torch.zeros(1, 64, self.ny, self.nx, device=device)
        xi = pillar_coords[:, 0]
        yi = pillar_coords[:, 1]
        pseudo[0, :, yi, xi] = vfe_out.T

        # Backbone + neck
        multi_scale = self.backbone(pseudo)            # list of 3 feature maps
        bev         = self.neck(multi_scale)           # [1, 256, ny/2, nx/2]

        # Upsample to match cam_bev spatial size
        if bev.shape[-2:] != self.out_spatial_size:
            bev = torch.nn.functional.interpolate(
                bev, size=self.out_spatial_size,
                mode='bilinear', align_corners=False,
            )
        return bev   # [1, 256, 128, 128]