"""
map_labels.py - nuScenes BEV map label rasterisation.

Converts nuScenes vector map annotations (lane_divider, ped_crossing,
road_segment) into binary BEV masks matching the camera/LiDAR BEV grid.

The output is a [3, H, W] uint8 tensor with one channel per class:
    channel 0 — lane_divider
    channel 1 — ped_crossing
    channel 2 — road_segment

Used as supervision target for the segmentation head in the end-to-end
alignment training loop.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


LAYER_NAMES = ['lane_divider', 'ped_crossing', 'road_segment']
NUM_CLASSES  = len(LAYER_NAMES)

ALL_LOCATIONS = [
    'singapore-onenorth',
    'boston-seaport',
    'singapore-queenstown',
    'singapore-hollandvillage',
]


def load_nusc_maps(dataroot: str) -> Dict[str, NuScenesMap]:
    """
    Pre-load all four NuScenesMap instances.

    Returns
    -------
    dict mapping location name -> NuScenesMap
    """
    return {
        loc: NuScenesMap(dataroot=dataroot, map_name=loc)
        for loc in ALL_LOCATIONS
    }


def get_ego_pose_and_location(nusc: NuScenes, sample_token: str):
    """
    Return ego position, yaw (degrees), and map location for a sample.

    Uses the LIDAR_TOP ego pose as the reference frame, consistent with
    the LiDAR BEV coordinate system.

    Returns
    -------
    x, y     : float - ego position in global map coordinates (metres)
    yaw_deg  : float - ego heading in degrees
    location : str   - map name (e.g. 'singapore-onenorth')
    """
    sample   = nusc.get('sample', sample_token)
    scene    = nusc.get('scene',  sample['scene_token'])
    log      = nusc.get('log',    scene['log_token'])
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    x, y    = ego_pose['translation'][0], ego_pose['translation'][1]
    q       = Quaternion(ego_pose['rotation'])
    yaw_deg = float(np.degrees(q.yaw_pitch_roll[0]))

    return x, y, yaw_deg, log['location']


def get_bev_map_labels(
    nusc: NuScenes,
    nusc_maps: Dict[str, NuScenesMap],
    sample_token: str,
    bev_extent: float = 51.2,
    canvas_size: tuple = (128, 128),
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Rasterise nuScenes map elements onto a BEV grid for a given sample.

    Parameters
    ----------
    nusc         : NuScenes instance
    nusc_maps    : dict from load_nusc_maps()
    sample_token : nuScenes sample token
    bev_extent   : half-width of BEV in metres (default 51.2 → ±51.2m)
    canvas_size  : (H, W) of output grid - must match cam_bev spatial dims
    device       : torch device for output tensor

    Returns
    -------
    labels : FloatTensor [NUM_CLASSES, H, W] with values in {0.0, 1.0}
             channel 0 = lane_divider
             channel 1 = ped_crossing
             channel 2 = road_segment
    """
    x, y, yaw_deg, location = get_ego_pose_and_location(nusc, sample_token)

    nusc_map = nusc_maps[location]

    # patch_box: (centre_x, centre_y, height_m, width_m)
    patch_size = bev_extent * 2   # 102.4m
    patch_box  = (x, y, patch_size, patch_size)

    mask = nusc_map.get_map_mask(
        patch_box,
        yaw_deg,
        LAYER_NAMES,
        canvas_size=canvas_size,
    )   # [NUM_CLASSES, H, W] uint8

    return torch.from_numpy(mask.astype(np.float32)).to(device)