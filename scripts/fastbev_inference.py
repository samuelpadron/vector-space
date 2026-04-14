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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torchvision.models import resnet50
from torchvision.transforms.functional import normalize
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class CustomFPN(nn.Module):
    """CustomFPN matching original FastBEV architecture."""
    def __init__(self, in_channels=[1024, 2048], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            # Match original: lateral_convs.X.conv
            lateral = nn.Sequential()
            lateral.add_module('conv', nn.Conv2d(in_ch, out_channels, 1))
            self.lateral_convs.append(lateral)

            # Match original: fpn_convs.X.conv
            fpn = nn.Sequential()
            fpn.add_module('conv', nn.Conv2d(out_channels, out_channels, 3, padding=1))
            self.fpn_convs.append(fpn)

    def forward(self, features):
        # features: list of feature maps from backbone
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')

        # Output convolutions
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs[0]  # Return finest level


class FastrayTransformer(nn.Module):
    """
    FastBEV's ray-based image-to-BEV transformation.
    """
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        image_size: Tuple[int, int] = (256, 704),
        feature_size: Tuple[int, int] = (16, 44),
        grid_config: Dict = None,
        stride: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.stride = stride

        # Default grid config
        if grid_config is None:
            grid_config = {
                'x': [-51.2, 51.2, 0.8],
                'y': [-51.2, 51.2, 0.8],
                'z': [-2.5, 4.5, 1.0],
                'depth': [1.0, 60.0, 1.0],
            }
        self.grid_config = grid_config

        # Grid sizes
        self.X = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])  # 128
        self.Y = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])  # 128
        self.Z = int((grid_config['z'][1] - grid_config['z'][0]) / grid_config['z'][2])  # 7
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])  # 59

        # Create grid info
        self.grid_lower_bound = torch.tensor([grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]])
        self.grid_interval = torch.tensor([grid_config['x'][2], grid_config['y'][2], grid_config['z'][2]])

        # Depth + feature network (D + out_channels outputs)
        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, kernel_size=1, padding=0)

        # Create voxel coordinates
        self.register_buffer('voxel_coords', self._create_voxel_coords())

    def _create_voxel_coords(self):
        """Create 3D voxel coordinates in ego frame."""
        x = torch.arange(self.X).view(-1, 1, 1).expand(-1, self.Y, self.Z).float()
        y = torch.arange(self.Y).view(1, -1, 1).expand(self.X, -1, self.Z).float()
        z = torch.arange(self.Z).view(1, 1, -1).expand(self.X, self.Y, -1).float()
        coords = torch.stack((x, y, z), dim=3)
        coords = coords * self.grid_interval + self.grid_lower_bound
        coords = coords.reshape(-1, 3)
        return coords

    def forward(self, img_feats, cam2ego, cam_intrinsics, img_aug_matrix=None):
        """
        Project image features to BEV.
        """
        B, N, C, H, W = img_feats.shape
        device = img_feats.device

        # Apply depth network
        x = img_feats.view(B * N, C, H, W)
        x = self.depth_net(x)
        x = x.view(B, N, self.D + self.out_channels, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, N, H, W, D+C)

        # Split depth and features
        depth = x[..., :self.D].softmax(dim=-1)  # (B, N, H, W, D)
        feat = x[..., self.D:]  # (B, N, H, W, C)

        # Project voxels to images and sample features
        bev_feat = self._project_and_sample(feat, depth, cam2ego, cam_intrinsics, img_aug_matrix)

        return bev_feat, depth

    def _project_and_sample(self, feat, depth, cam2ego, cam_intrinsics, img_aug_matrix):
        """Project voxel coordinates to images and sample features using vectorized ops."""
        B, N, H, W, C = feat.shape
        device = feat.device

        # Initialize BEV feature volume
        bev_feat = torch.zeros(B, self.X, self.Y, self.Z, C, device=device, dtype=feat.dtype)

        # Get voxel coordinates
        voxel_coords = self.voxel_coords.to(device)  # (num_voxels, 3)
        num_voxels = voxel_coords.shape[0]

        # Create depth values for each depth bin
        depth_bins = torch.arange(self.D, device=device).float() * self.grid_config['depth'][2] + self.grid_config['depth'][0]

        for b in range(B):
            for n in range(N):
                # Get camera parameters
                c2e = cam2ego[b, n]  # (4, 4)
                K = cam_intrinsics[b, n]  # (3, 3)

                # Transform voxels to camera frame
                e2c = torch.inverse(c2e)

                # Homogeneous voxel coords
                voxel_homo = torch.cat([voxel_coords, torch.ones(num_voxels, 1, device=device)], dim=1)

                # Transform to camera frame
                cam_coords = (e2c @ voxel_homo.T).T[:, :3]  # (num_voxels, 3)

                # Get depth values
                z = cam_coords[:, 2]
                valid_z = z > 0.5

                # Avoid division by zero
                z_safe = torch.clamp(z, min=0.1)

                # Project to image plane
                cam_coords_norm = cam_coords[:, :2] / z_safe.unsqueeze(-1)
                cam_coords_homo = torch.cat([cam_coords_norm, torch.ones(num_voxels, 1, device=device)], dim=1)
                img_coords = (K @ cam_coords_homo.T).T[:, :2]

                # Convert to feature map coordinates
                feat_coords = img_coords / self.stride

                # Check bounds
                valid_x = (feat_coords[:, 0] >= 0) & (feat_coords[:, 0] < W)
                valid_y = (feat_coords[:, 1] >= 0) & (feat_coords[:, 1] < H)
                valid = valid_x & valid_y & valid_z

                # Get depth bin
                depth_bin = ((z - self.grid_config['depth'][0]) / self.grid_config['depth'][2]).long()
                valid_depth = (depth_bin >= 0) & (depth_bin < self.D)
                valid = valid & valid_depth

                # Sample for valid points
                valid_idx = torch.where(valid)[0]
                if len(valid_idx) == 0:
                    continue

                # Get image coordinates
                u = feat_coords[valid_idx, 0].long().clamp(0, W-1)
                v = feat_coords[valid_idx, 1].long().clamp(0, H-1)
                d = depth_bin[valid_idx].clamp(0, self.D-1)

                # Sample features and depth weights
                sampled_feat = feat[b, n, v, u, :]
                sampled_depth = depth[b, n, v, u, d]

                # Weight features by depth
                weighted_feat = sampled_feat * sampled_depth.unsqueeze(-1)

                # Convert voxel index to 3D coordinates
                vx = valid_idx // (self.Y * self.Z)
                vy = (valid_idx % (self.Y * self.Z)) // self.Z
                vz = valid_idx % self.Z

                # Accumulate using scatter_add for efficiency
                flat_idx = vx * self.Y * self.Z + vy * self.Z + vz
                bev_flat = bev_feat[b].view(-1, C)
                bev_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), weighted_feat)

        # Collapse Z dimension (sum)
        bev_feat = bev_feat.sum(dim=3)  # (B, X, Y, C)
        bev_feat = bev_feat.permute(0, 3, 2, 1)  # (B, C, Y, X)

        return bev_feat


class BasicBlock(nn.Module):
    """Basic residual block matching mmdet implementation."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class CustomResNetBEV(nn.Module):
    """BEV encoder backbone matching original FastBEV."""
    def __init__(self, numC_input=64, num_channels=[128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_ch = numC_input

        for i, out_ch in enumerate(num_channels):
            # First block with stride 2 and downsample
            downsample = nn.Conv2d(curr_ch, out_ch, 3, stride=2, padding=1)
            block1 = BasicBlock(curr_ch, out_ch, stride=2, downsample=downsample)
            # Second block without downsample
            block2 = BasicBlock(out_ch, out_ch)

            self.layers.append(nn.Sequential(block1, block2))
            curr_ch = out_ch

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class FPN_LSS(nn.Module):
    """FPN neck for BEV features matching original FastBEV."""
    def __init__(self, in_channels=640, out_channels=256, scale_factor=4, extra_upsample=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2  # For extra_upsample

        # Main conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
        )

        # Extra upsample
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * channels_factor, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, padding=0),
        )

    def forward(self, feats):
        # feats: list of [128ch@64x64, 256ch@32x32, 512ch@16x16]
        x2 = feats[0]  # 128 channels, higher res
        x1 = feats[2]  # 512 channels, lower res

        x1 = self.up(x1)  # Upsample 512ch to match 128ch resolution
        x = torch.cat([x2, x1], dim=1)  # 640 channels
        x = self.conv(x)
        x = self.up2(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv + BN + optional ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class ConvModule(nn.Module):
    """Conv + BN matching mmcv ConvModule structure."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Sequential):
    """Separate head for each output type matching CenterHead structure.

    Checkpoint structure is:
    - task_heads.0.reg.0.conv/bn (ConvModule)
    - task_heads.0.reg.1 (final conv with bias)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            ConvModule(in_channels, in_channels, kernel_size, padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        )


class CenterHead(nn.Module):
    """CenterPoint-style detection head matching original FastBEV."""
    def __init__(self, in_channels=256, share_conv_channel=64, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Shared conv
        self.shared_conv = ConvBNReLU(in_channels, share_conv_channel, 3, padding=1)

        # Task heads (matching task_heads.0.X structure)
        self.task_heads = nn.ModuleList([
            nn.ModuleDict({
                'heatmap': SeparateHead(share_conv_channel, num_classes),
                'reg': SeparateHead(share_conv_channel, 2),
                'height': SeparateHead(share_conv_channel, 1),
                'dim': SeparateHead(share_conv_channel, 3),
                'rot': SeparateHead(share_conv_channel, 2),
                'vel': SeparateHead(share_conv_channel, 2),
            })
        ])

    def forward(self, x):
        x = self.shared_conv(x)

        outputs = []
        for task_head in self.task_heads:
            out = {}
            for name, head in task_head.items():
                out[name] = head(x)
            outputs.append(out)

        return outputs


class FastBEV(nn.Module):
    """
    FastBEV implementation matching pretrained checkpoint.
    """
    def __init__(
        self,
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    ):
        super().__init__()
        self.bev_channels = bev_channels

        # Backbone: ResNet50
        backbone = resnet50(weights=None)
        self.img_backbone = backbone

        # Neck
        self.img_neck = CustomFPN(in_channels=[1024, 2048], out_channels=in_channels)

        # View transformer
        self.img_view_transformer = FastrayTransformer(
            in_channels=in_channels,
            out_channels=bev_channels,
            image_size=image_size,
            feature_size=feature_size,
        )

        # BEV encoder backbone
        self.img_bev_encoder_backbone = CustomResNetBEV(
            numC_input=bev_channels,
            num_channels=[bev_channels * 2, bev_channels * 4, bev_channels * 8]
        )

        # BEV encoder neck
        # Input: 512 (from layer 2) + 128 (from layer 0) = 640
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=bev_channels * 8 + bev_channels * 2,
            out_channels=out_channels
        )

        # Detection head
        self.pts_bbox_head = CenterHead(in_channels=out_channels, num_classes=num_classes)

    def extract_img_feat(self, imgs):
        """Extract image features from all cameras."""
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)

        # Backbone (ResNet50)
        x = self.img_backbone.conv1(imgs)
        x = self.img_backbone.bn1(x)
        x = self.img_backbone.relu(x)
        x = self.img_backbone.maxpool(x)

        x1 = self.img_backbone.layer1(x)
        x2 = self.img_backbone.layer2(x1)
        x3 = self.img_backbone.layer3(x2)  # 1024 channels
        x4 = self.img_backbone.layer4(x3)  # 2048 channels

        # Neck
        feat = self.img_neck([x3, x4])

        _, C_out, H_out, W_out = feat.shape
        feat = feat.view(B, N, C_out, H_out, W_out)

        return feat

    def forward(self, imgs, cam2ego, cam_intrinsics, img_aug_matrix=None):
        """Forward pass."""
        # Extract image features
        img_feats = self.extract_img_feat(imgs)

        # Project to BEV
        bev_feat, depth = self.img_view_transformer(img_feats, cam2ego, cam_intrinsics, img_aug_matrix)

        # Encode BEV
        bev_feats = self.img_bev_encoder_backbone(bev_feat)
        bev_feat = self.img_bev_encoder_neck(bev_feats)

        # Detection head
        preds = self.pts_bbox_head(bev_feat)

        return {
            'predictions': preds,
            'bev_feat': bev_feat,
            'depth': depth,
        }


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load pretrained checkpoint with key remapping."""
    print(f"\nLoading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']

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

    cam_names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    images = []
    intrinsics = []
    cam2egos = []
    img_aug_matrices = []

    for cam_name in cam_names:
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
        images.append(img_tensor)

        # Get calibration
        intrinsic, cam2ego = get_sensor_transforms(nusc, cam_token)

        # Adjust intrinsics for resize
        scale_x = target_size[1] / orig_size[0]
        scale_y = target_size[0] / orig_size[1]
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, :] *= scale_x
        intrinsic_scaled[1, :] *= scale_y

        intrinsics.append(torch.from_numpy(intrinsic_scaled).float())
        cam2egos.append(torch.from_numpy(cam2ego).float())

        # Image augmentation matrix (identity for inference)
        img_aug = torch.eye(3)
        img_aug_matrices.append(img_aug)

    images = torch.stack(images, dim=0)
    intrinsics = torch.stack(intrinsics, dim=0)
    cam2egos = torch.stack(cam2egos, dim=0)
    img_aug_matrices = torch.stack(img_aug_matrices, dim=0)

    return images, intrinsics, cam2egos, img_aug_matrices, sample


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


def visualize_bev_with_detections(bev_feat, preds, save_path=None, input_images=None):
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
    """
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Decode predictions
    detections = decode_predictions(preds, score_threshold=0.2)
    print(f"  Found {len(detections)} detections above threshold")

    # BEV feature visualization - rotate 90° CCW so FRONT is UP
    bev = bev_feat[0].mean(dim=0).detach().cpu().numpy()
    bev = np.rot90(bev, k=1)  # Rotate 90° counter-clockwise

    # Heatmap - same rotation
    heatmap = preds[0]['heatmap'][0].sigmoid().max(dim=0)[0].detach().cpu().numpy()
    heatmap = np.rot90(heatmap, k=1)

    # Create figure with cameras on top, BEV on bottom
    if input_images is not None:
        fig = plt.figure(figsize=(18, 12))
        # Top row: 6 cameras (or fewer)
        # Bottom row: BEV features, heatmap, detections
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.5], hspace=0.25, wspace=0.1)

        # Camera images on top row
        cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(min(6, input_images.shape[0])):
            ax_cam = fig.add_subplot(gs[0, i])
            img = input_images[i].cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax_cam.imshow(img)
            ax_cam.set_title(cam_names[i], fontsize=9)
            ax_cam.axis('off')

        # BEV plots on bottom row (spanning 2 columns each)
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

        # Rotate yaw by 90° as well
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
    """Visualize camera images."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    for i in range(6):
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

def main():
    # Paths
    nuscenes_root = Path('./data/nuscenes')
    checkpoint_path = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
    output_dir = Path('./viz_output/fastbev')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating FastBEV model...")
    model = FastBEV(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )

    # Load pretrained weights
    if checkpoint_path.exists():
        model = load_checkpoint(model, checkpoint_path, device)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with random weights...")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load nuScenes
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(nuscenes_root), verbose=False)

    # Process multiple samples
    for sample_idx in range(min(3, len(nusc.sample))):
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']
        print(f"\nProcessing sample {sample_idx}: {sample_token[:8]}...")

        # Load data
        images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(nusc, sample_token)

        # Add batch dimension and move to device
        images = images.unsqueeze(0).to(device)
        intrinsics = intrinsics.unsqueeze(0).to(device)
        cam2egos = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        print(f"  Input shape: {images.shape}")

        # Run inference
        print("  Running inference...")
        with torch.no_grad():
            outputs = model(images, cam2egos, intrinsics, img_aug_matrices)

        print(f"  BEV features shape: {outputs['bev_feat'].shape}")
        print(f"  Heatmap shape: {outputs['predictions'][0]['heatmap'].shape}")

        # Visualize outputs with input images
        visualize_bev_with_detections(
            outputs['bev_feat'],
            outputs['predictions'],
            save_path=output_dir / f'detections_{sample_idx}.png',
            input_images=images[0]  # Pass the camera images
        )

    print(f"\nDone! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
