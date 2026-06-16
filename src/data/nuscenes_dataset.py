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
    Each item is a (curr, past_frames...) tuple from the same scene.

    prev_offsets controls which past frames are returned, e.g. [5, 10] yields
    frames at t-5 and t-10 alongside the current frame t.  Only samples that
    have at least max(prev_offsets) frames of history are included.

    Tensor shapes returned per item (monocam, N=1):
        img_curr / cam2ego_curr / intrinsics_curr : [1, 3, H, W] / [1, 4, 4] / [1, 3, 3]
        imgs_prev      : [T, 1, 3, H, W]   T = len(prev_offsets)
        cam2ego_prev   : [T, 1, 4, 4]
        intrinsics_prev: [T, 1, 3, 3]
        se2_list       : [T, 3]   SE(2) from each past frame → current (dx_grid, dy_grid, dyaw)
        gt_boxes       : list[dict]

    After collate_fn the leading batch dim is added:
        img_curr   : [B, 1, 3, H, W]
        imgs_prev  : [B, T, 1, 3, H, W]
        se2_list   : [B, T, 3]
    """

    def __init__(self, nusc, split='train', target_size=(256, 704), prev_offsets=(1,)):
        self.nusc         = nusc
        self.target_size  = target_size
        self.prev_offsets = list(prev_offsets)
        max_offset        = max(self.prev_offsets)

        valid_scenes = set(create_splits_scenes()[split])

        self.sample_tokens = []
        for s in nusc.sample:
            if nusc.get('scene', s['scene_token'])['name'] not in valid_scenes:
                continue
            # Walk back max_offset steps to verify sufficient history exists
            sample = s
            ok = True
            for _ in range(max_offset):
                if sample['prev'] == '':
                    ok = False
                    break
                sample = nusc.get('sample', sample['prev'])
            if ok:
                self.sample_tokens.append(s['token'])

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        token_curr  = self.sample_tokens[idx]
        sample_curr = self.nusc.get('sample', token_curr)

        img_curr, intr_curr, c2e_curr, _, ego_curr, _ = load_sample(
            self.nusc, token_curr, self.target_size
        )

        # Walk the prev chain once and cache every token up to max_offset
        max_offset = max(self.prev_offsets)
        token_chain = []   # token_chain[k] = token for t-(k+1)
        sample = sample_curr
        for _ in range(max_offset):
            token_chain.append(sample['prev'])
            sample = self.nusc.get('sample', sample['prev'])

        # Load each requested past frame
        imgs_prev, c2e_prev_list, intr_prev_list, se2s = [], [], [], []
        for offset in self.prev_offsets:
            token_p = token_chain[offset - 1]
            img_p, intr_p, c2e_p, _, ego_p, _ = load_sample(self.nusc, token_p, self.target_size)
            imgs_prev.append(img_p)
            c2e_prev_list.append(c2e_p)
            intr_prev_list.append(intr_p)
            se2s.append(_compute_se2(ego_p, ego_curr))

        gt_boxes = load_gt_boxes(self.nusc, token_curr)

        return {
            'idx':              torch.tensor(idx),
            'img_curr':         img_curr,                        # [1, 3, H, W]
            'cam2ego_curr':     c2e_curr,                        # [1, 4, 4]
            'intrinsics_curr':  intr_curr,                       # [1, 3, 3]
            'imgs_prev':        torch.stack(imgs_prev),          # [T, 1, 3, H, W]
            'cam2ego_prev':     torch.stack(c2e_prev_list),      # [T, 1, 4, 4]
            'intrinsics_prev':  torch.stack(intr_prev_list),     # [T, 1, 3, 3]
            'se2_list':         torch.stack(se2s),               # [T, 3]
            'gt_boxes':         gt_boxes,
        }


def collate_fn(batch):
    """Stack tensors; keep gt_boxes as a list (variable-length per sample)."""
    return {
        'idx':              torch.stack([b['idx']             for b in batch]),
        'img_curr':         torch.stack([b['img_curr']        for b in batch]),  # [B,1,3,H,W]
        'cam2ego_curr':     torch.stack([b['cam2ego_curr']    for b in batch]),  # [B,1,4,4]
        'intrinsics_curr':  torch.stack([b['intrinsics_curr'] for b in batch]),  # [B,1,3,3]
        'imgs_prev':        torch.stack([b['imgs_prev']       for b in batch]),  # [B,T,1,3,H,W]
        'cam2ego_prev':     torch.stack([b['cam2ego_prev']    for b in batch]),  # [B,T,1,4,4]
        'intrinsics_prev':  torch.stack([b['intrinsics_prev'] for b in batch]),  # [B,T,1,3,3]
        'se2_list':         torch.stack([b['se2_list']        for b in batch]),  # [B,T,3]
        'gt_boxes':         [b['gt_boxes'] for b in batch],
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

    yaw_prev = q_prev.yaw_pitch_roll[0]
    dyaw = q_curr.yaw_pitch_roll[0] - yaw_prev
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi   # normalise to [-π, π]

    dx_ego =  np.cos(yaw_prev) * dx_m + np.sin(yaw_prev) * dy_m
    dy_ego = -np.sin(yaw_prev) * dx_m + np.cos(yaw_prev) * dy_m

    return torch.tensor(
        [dx_ego / GRID_RESOLUTION, dy_ego / GRID_RESOLUTION, dyaw],
        dtype=torch.float32,
    )