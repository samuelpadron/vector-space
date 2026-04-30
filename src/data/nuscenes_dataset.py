"""
NuScenes dataset for FastBEV4D temporal fusion training.
Yields consecutive frame pairs within the same scene.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_scenes

from .nuscenes_loader import load_sample, load_gt_boxes

GRID_RESOLUTION = 0.8   # metres per BEV cell
GRID_RANGE      = 51.2  # BEV extent [-GRID_RANGE, GRID_RANGE] metres


class NuScenesSequenceDataset(Dataset):
    """
    Each item is a consecutive (prev, curr) frame pair from the same scene,
    with the SE(2) ego-motion transform between them.

    First frames of each scene (no prev) are excluded so every item has
    a valid previous frame.

    Tensor shapes returned (monocam, N=1):
        img_curr / img_prev        : [1, 3, H, W]
        cam2ego_curr / cam2ego_prev: [1, 4, 4]
        intrinsics_curr / _prev    : [1, 3, 3]
        se2                        : [3]
        gt_boxes                   : list[dict]

    After collate_fn the leading batch dim is added, giving:
        img_*       : [B, 1, 3, H, W]
        cam2ego_*   : [B, 1, 4, 4]
        intrinsics_*: [B, 1, 3, 3]

    The N=1 dimension is kept (not squeezed) so the model's N-camera
    API works without modification for both mono and surround-view.
    """

    def __init__(self, nusc, split='train', target_size=(256, 704)):
        self.nusc        = nusc
        self.target_size = target_size

        valid_scenes = set(create_splits_scenes()[split])

        self.sample_tokens = [
            s['token'] for s in nusc.sample
            if s['prev'] != ''
            and nusc.get('scene', s['scene_token'])['name'] in valid_scenes
        ]

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        token_curr  = self.sample_tokens[idx]
        sample_curr = self.nusc.get('sample', token_curr)
        token_prev  = sample_curr['prev']

        img_curr, intr_curr, c2e_curr, _, ego_curr, _ = load_sample(
            self.nusc, token_curr, self.target_size
        )
        img_prev, intr_prev, c2e_prev, _, ego_prev, _ = load_sample(
            self.nusc, token_prev, self.target_size
        )

        # load_sample already returns [N, ...] tensors (N=1 for monocam).
        # Do NOT squeeze — the model expects the N dimension to be present.
        # Shapes here: img [1,3,H,W], c2e [1,4,4], intr [1,3,3]

        se2      = _compute_se2(ego_prev, ego_curr)
        gt_boxes = load_gt_boxes(self.nusc, token_curr)

        return {
            'idx':             idx,
            'img_curr':        img_curr,    # [1, 3, H, W]
            'cam2ego_curr':    c2e_curr,    # [1, 4, 4]
            'intrinsics_curr': intr_curr,   # [1, 3, 3]
            'img_prev':        img_prev,    # [1, 3, H, W]
            'cam2ego_prev':    c2e_prev,    # [1, 4, 4]
            'intrinsics_prev': intr_prev,   # [1, 3, 3]
            'se2':             se2,         # [3]
            'gt_boxes':        gt_boxes,    # list[dict]
        }


def collate_fn(batch):
    """Stack tensors; keep gt_boxes as a list (variable-length per sample)."""
    return {
        'idx':             torch.stack([b['idx']             for b in batch]),
        'img_curr':        torch.stack([b['img_curr']        for b in batch]),  # [B,1,3,H,W]
        'cam2ego_curr':    torch.stack([b['cam2ego_curr']    for b in batch]),  # [B,1,4,4]
        'intrinsics_curr': torch.stack([b['intrinsics_curr'] for b in batch]),  # [B,1,3,3]
        'img_prev':        torch.stack([b['img_prev']        for b in batch]),  # [B,1,3,H,W]
        'cam2ego_prev':    torch.stack([b['cam2ego_prev']    for b in batch]),  # [B,1,4,4]
        'intrinsics_prev': torch.stack([b['intrinsics_prev'] for b in batch]),  # [B,1,3,3]
        'se2':             torch.stack([b['se2']             for b in batch]),  # [B,3]
        'gt_boxes':        [b['gt_boxes'] for b in batch],
    }


def _compute_se2(ego_pose_prev: dict, ego_pose_curr: dict) -> torch.Tensor:
    """
    SE(2) between two raw nuScenes ego_pose dicts.
    Returns [3] float tensor: (dx_grid, dy_grid, dyaw_rad).
    """
    t_prev = np.array(ego_pose_prev['translation'])
    t_curr = np.array(ego_pose_curr['translation'])
    q_prev = Quaternion(ego_pose_prev['rotation'])
    q_curr = Quaternion(ego_pose_curr['rotation'])

    dx_m = t_curr[0] - t_prev[0]
    dy_m = t_curr[1] - t_prev[1]

    dyaw = q_curr.yaw_pitch_roll[0] - q_prev.yaw_pitch_roll[0]
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi   # normalise to [-π, π]

    return torch.tensor(
        [dx_m / GRID_RESOLUTION, dy_m / GRID_RESOLUTION, dyaw],
        dtype=torch.float32,
    )