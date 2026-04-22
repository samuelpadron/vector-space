"""
nuScenes data loading utilities.
Handles camera sample loading, sensor calibration, checkpoint loading,
and detection decoding.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from torchvision.transforms.functional import normalize

# Add src to path for ego_motion import
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.ego_motion import EgoMotionEstimator, EgoPose


def get_sensor_transforms(nusc: NuScenes, sample_data_token: str):
    """
    Return (intrinsic 3×3, cam2ego 4×4) for a given sample_data token.
    """
    sd = nusc.get('sample_data', sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    intrinsic = np.array(cs['camera_intrinsic'])

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.array(cs['translation'])

    return intrinsic, cam2ego


def get_ego_pose(nusc: NuScenes, sample_token: str) -> EgoPose:
    """
    Extract ego pose (position and rotation) for a nuScenes sample.

    Args:
        nusc: NuScenes instance
        sample_token: Sample token

    Returns:
        EgoPose with x, y, z, roll, pitch, yaw
    """
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose_dict = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    t = np.array(ego_pose_dict['translation'])
    q = Quaternion(ego_pose_dict['rotation'])
    roll, pitch, yaw = q.yaw_pitch_roll  # yaw_pitch_roll returns (yaw, pitch, roll)
    # Reorder to (roll, pitch, yaw)
    roll, pitch, yaw = yaw, pitch, roll

    return EgoPose(x=t[0], y=t[1], z=t[2], roll=roll, pitch=pitch, yaw=yaw)


def compute_se2_transform(
    nusc: NuScenes,
    sample_token_prev: str,
    sample_token_curr: str,
    grid_size_m: float = 51.2,
    grid_resolution: float = 0.8,
) -> Optional[Dict]:
    """
    Compute SE(2) ego-motion transform between two consecutive samples.

    Args:
        nusc: NuScenes instance
        sample_token_prev: Previous sample token
        sample_token_curr: Current sample token
        grid_size_m: BEV grid extent
        grid_resolution: Meters per BEV cell

    Returns:
        Dict with 'dx', 'dy', 'dyaw' in grid units, or None if prev sample unavailable
    """
    pose_prev = get_ego_pose(nusc, sample_token_prev)
    pose_curr = get_ego_pose(nusc, sample_token_curr)

    estimator = EgoMotionEstimator(grid_size_m=grid_size_m, grid_resolution=grid_resolution)
    se2 = estimator.estimate_from_ego_pose(pose_prev, pose_curr)

    return {
        'dx': se2.dx,
        'dy': se2.dy,
        'dyaw': se2.dyaw,
    }



def load_sample(
    nusc: NuScenes,
    sample_token: str,
    target_size: Tuple[int, int] = (256, 704),
):
    """
    Load camera image and calibration data for a nuScenes sample (monocam: CAM_FRONT only).

    Returns
    -------
    images          : FloatTensor [1, 3, H, W]  (ImageNet-normalised, monocam)
    intrinsics      : FloatTensor [1, 3, 3]
    cam2egos        : FloatTensor [1, 4, 4]
    img_aug_matrices: FloatTensor [1, 3, 3]  (identity — inference mode)
    ego_pose        : dict with 'translation' and 'rotation' keys
    sample          : raw nuScenes sample dict
    """
    sample = nusc.get('sample', sample_token)

    # Use only CAM_FRONT for monocam setup
    cam_name = 'CAM_FRONT'
    cam_token = sample['data'][cam_name]
    cam_data = nusc.get('sample_data', cam_token)

    img = Image.open(Path(nusc.dataroot) / cam_data['filename']).convert('RGB')
    orig_w, orig_h = img.size

    img_resized = img.resize((target_size[1], target_size[0]))
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    intrinsic, cam2ego = get_sensor_transforms(nusc, cam_token)

    # Scale intrinsics to match resized image
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, :] *= target_size[1] / orig_w
    intrinsic_scaled[1, :] *= target_size[0] / orig_h

    # Get ego pose in dict format (translation + rotation)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    return (
        img_tensor.unsqueeze(0),  # [1, 3, H, W] for monocam
        torch.from_numpy(intrinsic_scaled).float().unsqueeze(0),  # [1, 3, 3]
        torch.from_numpy(cam2ego).float().unsqueeze(0),  # [1, 4, 4]
        torch.eye(3).unsqueeze(0),  # [1, 3, 3] for monocam
        ego_pose,  # Raw dict from nuScenes
        sample,
    )


def decode_predictions(preds, score_threshold: float = 0.3, max_objects: int = 50):
    """
    Decode CenterHead predictions to a list of bounding box dicts.

    Parameters
    ----------
    preds           : list of task dicts from CenterHead.forward()
    score_threshold : minimum heatmap score to keep a detection
    max_objects     : maximum detections returned (sorted by score)
    """
    task_preds = preds[0]

    heatmap = task_preds['heatmap'][0].sigmoid()   # (num_classes, H, W)
    reg     = task_preds['reg'][0]                 # (2, H, W)
    height  = task_preds['height'][0]              # (1, H, W)
    dim     = task_preds['dim'][0]                 # (3, H, W)
    rot     = task_preds['rot'][0]                 # (2, H, W)
    vel     = task_preds['vel'][0]                 # (2, H, W)

    num_classes, H, W = heatmap.shape

    # Simplified NMS via local max-pool
    heatmap_max = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    keep = (heatmap == heatmap_max) & (heatmap >= score_threshold)

    detections = []
    voxel_size = 0.8

    for cls in range(num_classes):
        y_idx, x_idx = torch.where(keep[cls])
        for y, x in zip(y_idx, x_idx):
            y, x = y.item(), x.item()
            score = heatmap[cls, y, x].item()

            import numpy as _np
            detections.append({
                'class':  cls,
                'score':  score,
                'x':      (x + reg[0, y, x].item()) * voxel_size - 51.2,
                'y':      (y + reg[1, y, x].item()) * voxel_size - 51.2,
                'z':      height[0, y, x].item(),
                'w':      _np.exp(dim[0, y, x].item()),
                'l':      _np.exp(dim[1, y, x].item()),
                'h':      _np.exp(dim[2, y, x].item()),
                'yaw':    _np.arctan2(rot[0, y, x].item(), rot[1, y, x].item()),
            })

    return sorted(detections, key=lambda d: d['score'], reverse=True)[:max_objects]


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load a pretrained FastBEV checkpoint with automatic key remapping.
    Prints a loading summary and returns the model with weights applied.
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    model_dict = model.state_dict()

    prefixes = (
        'img_backbone.',
        'img_neck.',
        'img_view_transformer.',
        'img_bev_encoder_backbone.',
        'img_bev_encoder_neck.',
        'pts_bbox_head.',
    )

    mapped_dict, unmatched = {}, []
    for ckpt_key, ckpt_val in state_dict.items():
        if not any(ckpt_key.startswith(p) for p in prefixes):
            unmatched.append(ckpt_key)
            continue
        if ckpt_key in model_dict:
            if model_dict[ckpt_key].shape == ckpt_val.shape:
                mapped_dict[ckpt_key] = ckpt_val
            else:
                print(f"  Shape mismatch: {ckpt_key}  "
                      f"model={model_dict[ckpt_key].shape}  ckpt={ckpt_val.shape}")
        else:
            unmatched.append(ckpt_key)

    model.load_state_dict(mapped_dict, strict=False)

    print(f"  Loaded {len(mapped_dict)}/{len(state_dict)} checkpoint keys")
    print(f"  Model has {len(model_dict)} keys total")
    if unmatched:
        print(f"  Unmatched keys: {len(unmatched)}")
        for k in unmatched[:5]:
            print(f"    - {k}")

    return model


# nuScenes category → detection class name + index
_CAT_TO_CLS = {
    'vehicle.car':                  ('car',                   0),
    'vehicle.truck':                ('truck',                 1),
    'vehicle.construction':         ('construction_vehicle',  2),
    'vehicle.bus.rigid':            ('bus',                   3),
    'vehicle.bus.bendy':            ('bus',                   3),
    'vehicle.trailer':              ('trailer',               4),
    'movable_object.barrier':       ('barrier',               5),
    'vehicle.motorcycle':           ('motorcycle',            6),
    'vehicle.bicycle':              ('bicycle',               7),
    'human.pedestrian.adult':       ('pedestrian',            8),
    'human.pedestrian.child':       ('pedestrian',            8),
    'human.pedestrian.wheelchair':  ('pedestrian',            8),
    'human.pedestrian.stroller':    ('pedestrian',            8),
    'movable_object.trafficcone':   ('traffic_cone',          9),
}


def load_gt_boxes(nusc, sample_token: str) -> list:
    """
    Load ground truth bounding boxes for a nuScenes sample in ego frame.

    Transforms each annotation from global frame → ego frame so the boxes
    sit in the same coordinate system as the BEV grid (ego-centred, x=forward,
    y=left, 0.8m/pixel over ±51.2m).

    Returns
    -------
    List of dicts with keys: x, y, z, w, l, h, yaw, class, name
    Only returns categories that map to one of the 10 detection classes.
    """
    from pyquaternion import Quaternion as _Quaternion
    import numpy as _np

    sample   = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    # Ego pose in global frame
    t_ego = _np.array(ego_pose['translation'])
    R_ego = _Quaternion(ego_pose['rotation'])

    gt_boxes = []
    for ann_token in sample['anns']:
        ann      = nusc.get('sample_annotation', ann_token)
        cat_name = ann['category_name']

        if cat_name not in _CAT_TO_CLS:
            continue

        cls_name, cls_idx = _CAT_TO_CLS[cat_name]

        # Global position → ego frame
        xyz_global = _np.array(ann['translation'])
        xyz_ego    = R_ego.inverse.rotate(xyz_global - t_ego)

        # Global yaw → ego frame yaw
        ann_q   = _Quaternion(ann['rotation'])
        ego_q   = R_ego.inverse * ann_q
        yaw_ego = ego_q.yaw_pitch_roll[0]   # yaw around z-axis

        w, l, h = ann['size']   # nuScenes: [width, length, height]

        gt_boxes.append({
            'x':     float(xyz_ego[0]),   # forward
            'y':     float(xyz_ego[1]),   # left
            'z':     float(xyz_ego[2]),
            'w':     float(w),
            'l':     float(l),
            'h':     float(h),
            'yaw':   float(yaw_ego),
            'class': cls_idx,
            'name':  cls_name,
        })

    return gt_boxes