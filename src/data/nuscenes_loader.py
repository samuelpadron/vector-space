"""
nuScenes data loading utilities.
Handles camera sample loading, sensor calibration, checkpoint loading,
and detection decoding.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from torchvision.transforms.functional import normalize


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


def load_sample(
    nusc: NuScenes,
    sample_token: str,
    target_size: Tuple[int, int] = (256, 704),
):
    """
    Load all 6 camera images and calibration data for a nuScenes sample.

    Returns
    -------
    images          : FloatTensor [6, 3, H, W]  (ImageNet-normalised)
    intrinsics      : FloatTensor [6, 3, 3]
    cam2egos        : FloatTensor [6, 4, 4]
    img_aug_matrices: FloatTensor [6, 3, 3]  (identity — inference mode)
    sample          : raw nuScenes sample dict
    """
    sample = nusc.get('sample', sample_token)

    cam_names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
    ]

    images, intrinsics, cam2egos, img_aug_matrices = [], [], [], []

    for cam_name in cam_names:
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)

        img = Image.open(Path(nusc.dataroot) / cam_data['filename']).convert('RGB')
        orig_w, orig_h = img.size

        img_resized = img.resize((target_size[1], target_size[0]))
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        img_tensor = normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images.append(img_tensor)

        intrinsic, cam2ego = get_sensor_transforms(nusc, cam_token)

        # Scale intrinsics to match resized image
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, :] *= target_size[1] / orig_w
        intrinsic_scaled[1, :] *= target_size[0] / orig_h

        intrinsics.append(torch.from_numpy(intrinsic_scaled).float())
        cam2egos.append(torch.from_numpy(cam2ego).float())
        img_aug_matrices.append(torch.eye(3))

    return (
        torch.stack(images),
        torch.stack(intrinsics),
        torch.stack(cam2egos),
        torch.stack(img_aug_matrices),
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
