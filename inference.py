"""
FastBEV++ Inference Script with Pretrained Weights
Pure PyTorch implementation - no mmcv/mmdet dependencies.
Based on FastBEV++ paper: https://arxiv.org/abs/2512.08237
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, List, Dict

from data.nuscenes_dataset import _compute_se2

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision.models import resnet50
from torchvision.transforms.functional import normalize
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from src.modules import FastBEV4D, FastBEV


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load pretrained checkpoint with key remapping."""
    print(f"\nLoading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model']

    model_dict = model.state_dict()

    # Map checkpoint keys to model keys
    mapped_dict = {}
    unmatched_ckpt = []

    for ckpt_key, ckpt_val in state_dict.items():
        # Handle backbone
        if ckpt_key.startswith('img_backbone.'):
            model_key = ckpt_key  # Direct mapping
        # Handle neck
        elif ckpt_key.startswith('img_neck.'):
            model_key = ckpt_key  # Direct mapping
        # Handle view transformer
        elif ckpt_key.startswith('img_view_transformer.'):
            model_key = ckpt_key  # Direct mapping
        # Handle BEV encoder backbone
        elif ckpt_key.startswith('img_bev_encoder_backbone.'):
            model_key = ckpt_key  # Direct mapping
        # Handle BEV encoder neck
        elif ckpt_key.startswith('img_bev_encoder_neck.'):
            model_key = ckpt_key  # Direct mapping
        # Handle temporal component
        elif ckpt_key.startswith('temporal_fusion'):
            model_key = ckpt_key
        # Handle detection head
        elif ckpt_key.startswith('pts_bbox_head.'):
            model_key = ckpt_key  # Direct mapping
        else:
            unmatched_ckpt.append(ckpt_key)
            continue

        if model_key in model_dict:
            if model_dict[model_key].shape == ckpt_val.shape:
                mapped_dict[model_key] = ckpt_val
            else:
                print(f"  Shape mismatch: {model_key} model={model_dict[model_key].shape} ckpt={ckpt_val.shape}")
        else:
            unmatched_ckpt.append(ckpt_key)

    # Load matched weights
    model.load_state_dict(mapped_dict, strict=False)

    # Statistics
    matched = len(mapped_dict)
    total_model = len(model_dict)
    total_ckpt = len(state_dict)

    print(f"  Loaded {matched}/{total_ckpt} checkpoint keys")
    print(f"  Model has {total_model} keys total")

    if unmatched_ckpt:
        print(f"  Unmatched checkpoint keys: {len(unmatched_ckpt)}")
        for k in unmatched_ckpt[:5]:
            print(f"    - {k}")

    return model

def get_sensor_transforms(nusc, sample_data_token):
    """Get sensor calibration."""
    sd = nusc.get('sample_data', sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    # Camera intrinsics
    intrinsic = np.array(cs['camera_intrinsic'])

    # Camera to ego transform
    translation = np.array(cs['translation'])
    rotation = Quaternion(cs['rotation']).rotation_matrix

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = rotation
    cam2ego[:3, 3] = translation

    return intrinsic, cam2ego


def load_sample(nusc, sample_token, target_size=(256, 704)):
    """Load a nuScenes sample with all cameras."""
    sample = nusc.get('sample', sample_token)

    cam_name = 'CAM_FRONT'
    cam_token = sample['data'][cam_name]
    cam_data = nusc.get('sample_data', cam_token)

    # Load image
    img_path = Path(nusc.dataroot) / cam_data['filename']
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (W, H)

    # Resize
    img_resized = img.resize((target_size[1], target_size[0]))
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0

    # Normalize
    img_tensor = normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Get calibration
    intrinsic, cam2ego = get_sensor_transforms(nusc, cam_token)

    # Adjust intrinsics for resize
    scale_x = target_size[1] / orig_size[0]
    scale_y = target_size[0] / orig_size[1]
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, :] *= scale_x
    intrinsic_scaled[1, :] *= scale_y

    intrinsics = torch.from_numpy(intrinsic_scaled).float()
    cam2ego = torch.from_numpy(cam2ego).float()

    # Image augmentation matrix (identity for inference)
    img_aug = torch.eye(3)

    img_tensor  = img_tensor.unsqueeze(0)   # [1, 3, H, W]
    intrinsics  = intrinsics.unsqueeze(0)   # [1, 3, 3]
    cam2ego     = cam2ego.unsqueeze(0)      # [1, 4, 4]

    return img_tensor, intrinsics, cam2ego, img_aug, sample


def decode_predictions(preds, score_threshold=0.3, max_objects=50):
    """Decode detection predictions to bounding boxes."""
    task_preds = preds[0]  # First task

    heatmap = task_preds['heatmap'][0].sigmoid()  # (num_classes, H, W)
    reg = task_preds['reg'][0]  # (2, H, W)
    height = task_preds['height'][0]  # (1, H, W)
    dim = task_preds['dim'][0]  # (3, H, W)
    rot = task_preds['rot'][0]  # (2, H, W)
    vel = task_preds['vel'][0]  # (2, H, W)

    num_classes, H, W = heatmap.shape

    # Find local maxima (simplified NMS)
    heatmap_max = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    keep = (heatmap == heatmap_max) & (heatmap >= score_threshold)

    detections = []

    for cls in range(num_classes):
        cls_keep = keep[cls]
        if not cls_keep.any():
            continue

        # Get positions of detections
        y_idx, x_idx = torch.where(cls_keep)
        scores = heatmap[cls, y_idx, x_idx]

        for i in range(len(scores)):
            y, x = y_idx[i].item(), x_idx[i].item()
            score = scores[i].item()

            # Get offset
            offset_x = reg[0, y, x].item()
            offset_y = reg[1, y, x].item()

            # Convert to world coordinates
            # Grid config: x,y range [-51.2, 51.2] with 0.8 resolution
            # Output size is H=W=128 after all processing
            voxel_size = 0.8
            x_world = (x + offset_x) * voxel_size - 51.2
            y_world = (y + offset_y) * voxel_size - 51.2
            z_world = height[0, y, x].item()

            # Get dimensions
            w = dim[0, y, x].item()
            l = dim[1, y, x].item()
            h = dim[2, y, x].item()

            # Get rotation
            sin_yaw = rot[0, y, x].item()
            cos_yaw = rot[1, y, x].item()
            yaw = np.arctan2(sin_yaw, cos_yaw)

            detections.append({
                'class': cls,
                'score': score,
                'x': x_world,
                'y': y_world,
                'z': z_world,
                'w': np.exp(w),  # Dimensions are often log-encoded
                'l': np.exp(l),
                'h': np.exp(h),
                'yaw': yaw,
            })

    # Sort by score and keep top detections
    detections = sorted(detections, key=lambda d: d['score'], reverse=True)[:max_objects]
    return detections


def visualize_bev_with_detections(bev_feat, preds, save_path=None, input_images=None, depth_maps=None):
    """Visualize BEV features with detected bounding boxes.

    Orientation: FRONT is UP (like driving forward up the screen)
    - UP = Forward (front of car)
    - DOWN = Back
    - LEFT = Left side of car
    - RIGHT = Right side of car

    Args:
        bev_feat: BEV features tensor
        preds: Detection predictions
        save_path: Path to save visualization
        input_images: Optional (N, 3, H, W) tensor of input camera images
        depth_maps: Optional (N, H, W) tensor of depth maps from DepthAnythingV2
    """
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Decode predictions
    detections = decode_predictions(preds, score_threshold=0.2)
    print(f"  Found {len(detections)} detections above threshold")

    # BEV feature visualization - rotate 90° CW so FRONT is UP
    bev = bev_feat[0].mean(dim=0).detach().cpu().numpy()
    bev = np.rot90(bev, k=3)  # Rotate 90° clockwise

    # Heatmap - same rotation
    heatmap = preds[0]['heatmap'][0].sigmoid().max(dim=0)[0].detach().cpu().numpy()
    heatmap = np.rot90(heatmap, k=3)

    # Create figure with cameras on top, BEV on bottom
    if input_images is not None:
        # Handle both batched (B, C, H, W) and single image (C, H, W)
        if input_images.ndim == 4:
            # Batched: (B, C, H, W)
            num_cams = input_images.shape[0]
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.5], hspace=0.25, wspace=0.1)
            cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        else:
            # Single image: (C, H, W) - mono camera
            num_cams = 1
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.25, wspace=0.15)
            cam_names = ['CAM_FRONT']
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(num_cams):
            if input_images.ndim == 4:
                ax_cam = fig.add_subplot(gs[0, i])
            else:
                # Single camera: left side for camera
                ax_cam = fig.add_subplot(gs[0, 0])
            if input_images.ndim == 4:
                img = input_images[i].cpu() * std + mean
            else:
                img = input_images.cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax_cam.imshow(img)
            ax_cam.set_title(cam_names[i], fontsize=9)
            ax_cam.axis('off')

        # Add depth heatmap next to camera if available
        if depth_maps is not None and input_images is not None:
            if input_images.ndim == 3:  # Single camera
                ax_depth = fig.add_subplot(gs[0, 1])
                depth_img = depth_maps[0].cpu().numpy() if depth_maps.ndim == 3 else depth_maps.cpu().numpy()
                ax_depth.imshow(depth_img, cmap='plasma')
                ax_depth.set_title('DepthAnythingV2', fontsize=9)
                ax_depth.axis('off')

        # BEV plots on bottom row
        if num_cams == 1:
            # For mono camera, create 3 equal-width plots
            axes = [
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2]),
            ]
        else:
            axes = [
                fig.add_subplot(gs[1, 0:2]),
                fig.add_subplot(gs[1, 2:4]),
                fig.add_subplot(gs[1, 4:6]),
            ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Grid config: [-51.2, 51.2] meters with 0.8m resolution = 128 pixels
    VOXEL_SIZE = 0.8  # meters per pixel
    GRID_CENTER = 64  # ego position in pixels

    # Ego vehicle marker for all plots (center of grid)
    ego_x, ego_y = GRID_CENTER, GRID_CENTER
    # Arrow points UP now (forward direction)
    ego_arrow_params = dict(head_width=2, head_length=1.5, fc='cyan', ec='white', linewidth=1, zorder=11)

    def add_distance_rings(ax, distances=[10, 20, 30, 40, 50]):
        """Add distance rings around ego vehicle."""
        for dist in distances:
            radius_px = dist / VOXEL_SIZE
            circle = plt.Circle((ego_x, ego_y), radius_px, fill=False,
                               color='white', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.add_patch(circle)
            # Label on right side of ring
            ax.text(ego_x + radius_px + 1, ego_y, f'{dist}m',
                   color='white', fontsize=7, alpha=0.7, va='center')

    def add_direction_labels(ax):
        """Add cardinal direction labels - FRONT is UP."""
        offset = 58
        ax.text(ego_x, ego_y + offset, 'FRONT', color='lime', fontsize=9,
               ha='center', va='bottom', alpha=0.9, fontweight='bold')
        ax.text(ego_x, ego_y - offset, 'BACK', color='white', fontsize=8,
               ha='center', va='top', alpha=0.7, fontweight='bold')
        ax.text(ego_x - offset, ego_y, 'LEFT', color='white', fontsize=8,
               ha='right', va='center', alpha=0.7, fontweight='bold')
        ax.text(ego_x + offset, ego_y, 'RIGHT', color='white', fontsize=8,
               ha='left', va='center', alpha=0.7, fontweight='bold')

    # BEV features
    axes[0].imshow(bev, cmap='viridis', origin='lower')
    axes[0].add_patch(plt.Circle((ego_x, ego_y), 3, color='cyan', fill=True, zorder=10))
    axes[0].arrow(ego_x, ego_y, 0, 8, **ego_arrow_params)  # Arrow points UP
    add_distance_rings(axes[0])
    axes[0].set_title('BEV Features (102.4m × 102.4m)')
    axes[0].set_xlabel('← Left | Right →')
    axes[0].set_ylabel('← Back | Front →')

    # Heatmap
    axes[1].imshow(heatmap, cmap='hot', origin='lower')
    axes[1].add_patch(plt.Circle((ego_x, ego_y), 3, color='cyan', fill=True, zorder=10))
    axes[1].arrow(ego_x, ego_y, 0, 8, **ego_arrow_params)  # Arrow points UP
    add_distance_rings(axes[1])
    axes[1].set_title('Detection Heatmap')
    axes[1].set_xlabel('← Left | Right →')
    axes[1].set_ylabel('← Back | Front →')

    # Detections in BEV
    H, W = bev.shape
    axes[2].imshow(np.zeros((H, W, 3)) + 0.1, origin='lower')

    # Add distance rings and direction labels
    add_distance_rings(axes[2])
    add_direction_labels(axes[2])

    # Draw ego vehicle at center - car shape pointing UP
    ego_l, ego_w = 4.5 / VOXEL_SIZE, 2.0 / VOXEL_SIZE  # ~4.5m x 2m car
    # Car pointing UP: length along Y, width along X
    ego_corners = np.array([
        [-ego_w/2, ego_l/2],    # Front left
        [ego_w/2, ego_l/2],     # Front right
        [ego_w/2, -ego_l/2],    # Back right
        [-ego_w/2, -ego_l/2],   # Back left
        [-ego_w/2, ego_l/2],    # Close
    ])
    ego_corners[:, 0] += ego_x
    ego_corners[:, 1] += ego_y
    axes[2].fill(ego_corners[:, 0], ego_corners[:, 1], color='cyan', alpha=0.7)
    axes[2].plot(ego_corners[:, 0], ego_corners[:, 1], color='white', linewidth=2)
    axes[2].text(ego_x, ego_y, 'EGO', fontsize=8, ha='center', va='center',
                color='black', fontweight='bold')
    # Direction arrow pointing UP
    axes[2].arrow(ego_x, ego_y + ego_l/2, 0, 3, head_width=1.5, head_length=1,
                 fc='lime', ec='white', linewidth=1.5, zorder=12)

    for det in detections:
        # Convert world coords to rotated pixel coords
        # Original: X=forward, Y=left
        # Rotated: X=right (neg left), Y=forward
        # So: new_x = -old_y, new_y = old_x (90° CCW rotation)
        world_x, world_y = det['x'], det['y']
        # Convert to pixels in rotated frame
        px = (-world_y + 51.2) / VOXEL_SIZE  # -Y becomes X
        py = (world_x + 51.2) / VOXEL_SIZE   # X becomes Y

        # Draw box
        w_px = det['w'] / VOXEL_SIZE
        l_px = det['l'] / VOXEL_SIZE

        color = class_colors[det['class']]

        # Rotate yaw by 90 well
        rotated_yaw = det['yaw'] + np.pi/2

        cos_yaw = np.cos(rotated_yaw)
        sin_yaw = np.sin(rotated_yaw)

        # Box corners (length along forward direction)
        corners = np.array([
            [-w_px/2, l_px/2],
            [w_px/2, l_px/2],
            [w_px/2, -l_px/2],
            [-w_px/2, -l_px/2],
            [-w_px/2, l_px/2],
        ])

        # Rotate
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rot_matrix.T
        corners[:, 0] += px
        corners[:, 1] += py

        axes[2].plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)

        # Distance from ego
        dist = np.sqrt(det['x']**2 + det['y']**2)
        axes[2].text(px, py + 4, f"{class_names[det['class']][:3]}",
                    fontsize=7, ha='center', va='bottom', color='white', fontweight='bold')
        axes[2].text(px, py - 4, f"{dist:.0f}m",
                    fontsize=6, ha='center', va='top', color='yellow')

    axes[2].set_xlim(0, W)
    axes[2].set_ylim(0, H)
    axes[2].set_title(f'Detected Objects ({len(detections)}) - 0.8m/pixel')
    axes[2].set_xlabel('← Left | Right →')
    axes[2].set_ylabel('← Back | Front →')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved BEV visualization to {save_path}")
    plt.close()


def visualize_cameras(images, save_path=None):
    """Visualize camera images. Supports both multi-camera (B, C, H, W) and mono (C, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    # Handle mono vs multi-camera
    if images.ndim == 3:
        # Mono camera: (C, H, W)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        img = images * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title('CAM_FRONT')
        ax.axis('off')
    else:
        # Multi-camera: (B, C, H, W)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(min(6, images.shape[0])):
            img = images[i] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(cam_names[i])
            axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved camera visualization to {save_path}")
    plt.close()

def visualize_comparison(
    image_curr,
    image_prev,
    out_baseline,
    out_4d,
    save_path=None,
):
    """
    6-panel comparison figure.

    Row 0: [CAM_FRONT curr] [baseline detections] [baseline heatmap]
    Row 1: [CAM_FRONT t-1]  [FastBEV4D detections] [FastBEV4D heatmap]
    """
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    VOXEL_SIZE  = 0.8
    GRID_CENTER = 64

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- helpers ---
    def add_rings(ax):
        for dist in [10, 20, 30, 40, 50]:
            r = dist / VOXEL_SIZE
            ax.add_patch(plt.Circle(
                (GRID_CENTER, GRID_CENTER), r,
                fill=False, color='white', linestyle='--', alpha=0.4, linewidth=0.8
            ))
            ax.text(GRID_CENTER + r + 1, GRID_CENTER, f'{dist}m',
                color='white', fontsize=7, alpha=0.6, va='center')

    def add_ego(ax):
        ax.add_patch(plt.Circle(
            (GRID_CENTER, GRID_CENTER), 3,
            color='cyan', fill=True, zorder=10
        ))
        ax.arrow(
            GRID_CENTER, GRID_CENTER, 0, 8,
            head_width=2, head_length=1.5,
            fc='cyan', ec='white', linewidth=1, zorder=11
        )

    def add_directions(ax):
        for txt, x, y, ha, va in [
            ('FRONT', GRID_CENTER,      GRID_CENTER + 58, 'center', 'bottom'),
            ('BACK',  GRID_CENTER,      GRID_CENTER - 58, 'center', 'top'),
            ('LEFT',  GRID_CENTER - 58, GRID_CENTER,      'right',  'center'),
            ('RIGHT', GRID_CENTER + 58, GRID_CENTER,      'left',   'center'),
        ]:
            ax.text(x, y, txt,
                color='lime' if txt == 'FRONT' else 'white',
                fontsize=8, ha=ha, va=va, alpha=0.8, fontweight='bold')

    def setup_bev_ax(ax, title):
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('← Left | Right →', fontsize=8)
        ax.set_ylabel('← Back | Front →', fontsize=8)

    def draw_detections(ax, preds):
        detections = decode_predictions(preds, score_threshold=0.2)
        for det in detections:
            px = (-det['y'] + 51.2) / VOXEL_SIZE
            py = ( det['x'] + 51.2) / VOXEL_SIZE
            w_px = det['w'] / VOXEL_SIZE
            l_px = det['l'] / VOXEL_SIZE
            color = class_colors[det['class']]
            rotated_yaw = det['yaw'] + np.pi / 2
            cos_y = np.cos(rotated_yaw)
            sin_y = np.sin(rotated_yaw)
            corners = np.array([
                [-w_px/2,  l_px/2],
                [ w_px/2,  l_px/2],
                [ w_px/2, -l_px/2],
                [-w_px/2, -l_px/2],
                [-w_px/2,  l_px/2],
            ])
            rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            corners = corners @ rot.T
            corners[:, 0] += px
            corners[:, 1] += py
            ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.5)
            dist = np.sqrt(det['x']**2 + det['y']**2)
            ax.text(px, py + 4, class_names[det['class']][:3],
                fontsize=6, ha='center', va='bottom', color='white', fontweight='bold')
            ax.text(px, py - 4, f'{dist:.0f}m',
                fontsize=6, ha='center', va='top', color='yellow')
        return len(detections)

    def show_camera(ax, img_tensor, title):
        if img_tensor is not None:
            img = img_tensor.cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, 'no frame available',
                ha='center', va='center',
                transform=ax.transAxes, color='white', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def show_heatmap(ax, preds, title):
        hm = preds[0]['heatmap'][0].sigmoid().max(dim=0)[0]
        hm = np.rot90(hm.detach().cpu().numpy(), k=3)
        ax.imshow(hm, cmap='hot', origin='lower')
        add_rings(ax)
        add_ego(ax)
        setup_bev_ax(ax, title)

    def show_detections(ax, preds, title):
        ax.imshow(np.zeros((128, 128, 3)) + 0.1, origin='lower')
        add_rings(ax)
        add_directions(ax)
        add_ego(ax)
        n = draw_detections(ax, preds)
        setup_bev_ax(ax, f'{title} ({n})')

    # --- ROW 0: baseline ---
    show_camera(axes[0, 0], image_curr, 'CAM_FRONT (t)')
    show_detections(axes[0, 1], out_baseline['predictions'], 'Baseline detections')
    show_heatmap(axes[0, 2], out_baseline['predictions'], 'Baseline heatmap')

    # --- ROW 1: FastBEV4D ---
    show_camera(axes[1, 0], image_prev, 'CAM_FRONT (t-1)')
    show_detections(axes[1, 1], out_4d['predictions'], 'FastBEV4D detections')
    show_heatmap(axes[1, 2], out_4d['predictions'], 'FastBEV4D heatmap')

    plt.suptitle('FastBEV baseline vs FastBEV4D (with temporal fusion)', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison to {save_path}")
    plt.close()
    
def main():
    # Paths
    nuscenes_root = Path('./data/nuscenes')
    checkpoint_path = Path('./checkpoints/fastbev4d_warmup/best.pth')
    output_dir = Path('./viz_output/fastbev4d')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Create model
    print("\nCreating FastBEV4D model...")
    model_4d = FastBEV4D(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )
    
    model_baseline = FastBEV(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )

    # Load pretrained weights
    if checkpoint_path.exists():
        model_4d = load_checkpoint(model_4d, checkpoint_path, device)
        model_baseline = load_checkpoint(model_baseline, checkpoint_path, device)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with random weights...")

    model_4d = model_4d.to(device)
    model_baseline = model_baseline.to(device)
    
    model_4d.eval()
    model_baseline.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model_4d.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load nuScenes
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(nuscenes_root), verbose=False)

    # Process multiple samples
    for sample_idx in range(min(3, len(nusc.sample))):
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']
        prev_token = nusc.get('sample', sample_token)['prev']
        print(f"\nProcessing sample {sample_idx}: {sample_token[:8]}...")

        # Load data
        images, intrinsics, cam2egos, img_aug_matrices, sample_data = load_sample(nusc, sample_token)

        # Compute depth map for visualization
        cam_name = 'CAM_FRONT'
        cam_token = sample_data['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)

        # Add batch dimension and move to device
        images = images.unsqueeze(0).to(device)
        intrinsics = intrinsics.unsqueeze(0).to(device)
        cam2egos = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        print(f"  Input shape: {images.shape}")

        # Run inference
        print("  Running inference...")
        with torch.no_grad():
            outputs_baseline = model_baseline(images, cam2egos, intrinsics, img_aug_matrices)
            
            if prev_token:
                # load prev frame
                imgs_prev, intr_prev, c2e_prev, _, _ = load_sample(nusc, prev_token)
                imgs_prev  = imgs_prev.unsqueeze(0).to(device)
                intr_prev  = intr_prev.unsqueeze(0).to(device)
                c2e_prev   = c2e_prev.unsqueeze(0).to(device)

                # get sparse BEV for prev frame
                img_feats_prev = model_4d.extract_img_feat(imgs_prev)
                bev_feat_prev, _ = model_4d.img_view_transformer(
                    img_feats_prev, c2e_prev, intr_prev
                )

                # compute SE2 between prev and curr ego poses
                ego_curr = nusc.get('ego_pose',
                    nusc.get('sample_data', sample_data['data']['CAM_FRONT'])['ego_pose_token'])
                ego_prev = nusc.get('ego_pose',
                    nusc.get('sample_data',
                        nusc.get('sample', prev_token)['data']['CAM_FRONT'])['ego_pose_token'])
                se2 = _compute_se2(ego_prev, ego_curr).unsqueeze(0).to(device)

                # FastBEV4D with prev frame
                outputs_4d = model_4d(
                    images, cam2egos, intrinsics,
                    bev_feat_prev=bev_feat_prev,
                    se2=se2,
                )

        print(f"  BEV features shape: {outputs_4d['bev_feat'].shape}")
        print(f"  Heatmap shape: {outputs_4d['predictions'][0]['heatmap'].shape}")

        # Visualize outputs with input images
        visualize_comparison(
            image_curr=images[0, 0],
            image_prev=imgs_prev[0, 0] if prev_token else None,
            out_baseline=out_baseline,
            out_4d=out_4d,
            save_path=output_dir / f'comparison_{sample_idx}.png',
        )
    print(f"\nDone! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
