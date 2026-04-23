"""
FastBEV4D: FastBEV++ with BEVDet4D-style temporal fusion.
Pure PyTorch, no mmcv/mmdet dependencies.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from torchvision.models import resnet50
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.temporal_fusion import BEVTemporalFusionConcat


class CustomFPN(nn.Module):
    """CustomFPN matching original FastBEV architecture."""
    def __init__(self, in_channels=[1024, 2048], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            lateral = nn.Sequential()
            lateral.add_module('conv', nn.Conv2d(in_ch, out_channels, 1))
            self.lateral_convs.append(lateral)

            fpn = nn.Sequential()
            fpn.add_module('conv', nn.Conv2d(out_channels, out_channels, 3, padding=1))
            self.fpn_convs.append(fpn)

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs[0]


class FastrayTransformer(nn.Module):
    """FastBEV ray-based image-to-BEV transformation with DepthAnythingV2."""
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

        if grid_config is None:
            grid_config = {
                'x': [-51.2, 51.2, 0.8],
                'y': [-51.2, 51.2, 0.8],
                'z': [-2.5, 4.5, 1.0],
                'depth': [1.0, 60.0, 1.0],
            }
        self.grid_config = grid_config

        self.X = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])
        self.Y = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])
        self.Z = int((grid_config['z'][1] - grid_config['z'][0]) / grid_config['z'][2])
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])

        self.grid_lower_bound = torch.tensor([grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]])
        self.grid_interval = torch.tensor([grid_config['x'][2], grid_config['y'][2], grid_config['z'][2]])

        self.feat_net = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

        self.register_buffer('voxel_coords', self._create_voxel_coords())

    def _create_voxel_coords(self):
        x = torch.arange(self.X).view(-1, 1, 1).expand(-1, self.Y, self.Z).float()
        y = torch.arange(self.Y).view(1, -1, 1).expand(self.X, -1, self.Z).float()
        z = torch.arange(self.Z).view(1, 1, -1).expand(self.X, self.Y, -1).float()
        coords = torch.stack((x, y, z), dim=3)
        coords = coords * self.grid_interval + self.grid_lower_bound
        return coords.reshape(-1, 3)

    def forward(self, img, img_feats, cam2ego, cam_intrinsics, img_aug_matrix=None):
        B, C, H, W = img_feats.shape
        device = img_feats.device

        feat = self.feat_net(img_feats)
        feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_denorm = (img * std + mean).clamp(0, 1)

        depth_maps = []
        for b_idx in range(B):
            img_pil = to_pil_image(img_denorm[b_idx].cpu())
            d = self.depth_pipeline(img_pil)["depth"]
            d_tensor = torch.from_numpy(np.array(d)).float()
            depth_maps.append(d_tensor)

        depth = torch.stack(depth_maps).to(device)
        depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        d_min = depth.flatten(1).min(1).values.view(B, 1, 1)
        d_max = depth.flatten(1).max(1).values.view(B, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        bev_feat = self._project_and_sample(feat, depth, cam2ego, cam_intrinsics, img_aug_matrix)
        return bev_feat, depth

    def _project_and_sample(self, feat, depth, cam2ego, cam_intrinsics, img_aug_matrix):
        B, H, W, C = feat.shape
        device = feat.device

        bev_feat = torch.zeros(B, self.X, self.Y, self.Z, C, device=device, dtype=feat.dtype)
        voxel_coords = self.voxel_coords.to(device)
        num_voxels = voxel_coords.shape[0]

        for b in range(B):
            K = cam_intrinsics[b]
            e2c = torch.inverse(cam2ego[b])

            voxel_homo = torch.cat([voxel_coords, torch.ones(num_voxels, 1, device=device)], dim=1)
            cam_coords = (e2c @ voxel_homo.mT).mT[:, :3]

            z = cam_coords[:, 2]
            valid_z = z > 0.5
            z_safe = torch.clamp(z, min=0.1)

            cam_coords_norm = cam_coords[:, :2] / z_safe.unsqueeze(-1)
            cam_coords_homo = torch.cat([cam_coords_norm, torch.ones(num_voxels, 1, device=device)], dim=1)
            img_coords = (K @ cam_coords_homo.T).T[:, :2]
            feat_coords = img_coords / self.stride

            valid_x = (feat_coords[:, 0] >= 0) & (feat_coords[:, 0] < W)
            valid_y = (feat_coords[:, 1] >= 0) & (feat_coords[:, 1] < H)
            valid = valid_x & valid_y & valid_z

            depth_bin = ((z - self.grid_config['depth'][0]) / self.grid_config['depth'][2]).long()
            valid_depth = (depth_bin >= 0) & (depth_bin < self.D)
            valid = valid & valid_depth

            valid_idx = torch.where(valid)[0]
            if len(valid_idx) == 0:
                continue

            u = feat_coords[valid_idx, 0].long().clamp(0, W-1)
            v = feat_coords[valid_idx, 1].long().clamp(0, H-1)

            sampled_feat = feat[b, v, u, :]
            sampled_depth = depth[b, v, u]
            weighted_feat = sampled_feat * sampled_depth.unsqueeze(-1)

            vx = valid_idx // (self.Y * self.Z)
            vy = (valid_idx % (self.Y * self.Z)) // self.Z
            vz = valid_idx % self.Z

            flat_idx = vx * self.Y * self.Z + vy * self.Z + vz
            bev_flat = bev_feat[b].view(-1, C)
            bev_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), weighted_feat)

        bev_feat = bev_feat.sum(dim=3)        # collapse Z
        bev_feat = bev_feat.permute(0, 3, 2, 1)  # (B, C, Y, X)
        return bev_feat


class BasicBlock(nn.Module):
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomResNetBEV(nn.Module):
    def __init__(self, numC_input=64, num_channels=[128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_ch = numC_input
        for out_ch in num_channels:
            downsample = nn.Conv2d(curr_ch, out_ch, 3, stride=2, padding=1)
            self.layers.append(nn.Sequential(
                BasicBlock(curr_ch, out_ch, stride=2, downsample=downsample),
                BasicBlock(out_ch, out_ch),
            ))
            curr_ch = out_ch

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class FPN_LSS(nn.Module):
    def __init__(self, in_channels=640, out_channels=256, scale_factor=4, extra_upsample=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, feats):
        x1 = self.up(feats[2])
        x = torch.cat([feats[0], x1], dim=1)
        return self.up2(self.conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu else x


class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            ConvModule(in_channels, in_channels, kernel_size, padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        )


class CenterHead(nn.Module):
    def __init__(self, in_channels=256, share_conv_channel=64, num_classes=10):
        super().__init__()
        self.shared_conv = ConvBNReLU(in_channels, share_conv_channel, 3, padding=1)
        self.task_heads = nn.ModuleList([
            nn.ModuleDict({
                'heatmap': SeparateHead(share_conv_channel, num_classes),
                'reg':     SeparateHead(share_conv_channel, 2),
                'height':  SeparateHead(share_conv_channel, 1),
                'dim':     SeparateHead(share_conv_channel, 3),
                'rot':     SeparateHead(share_conv_channel, 2),
                'vel':     SeparateHead(share_conv_channel, 2),
            })
        ])

    def forward(self, x):
        x = self.shared_conv(x)
        return [{name: head(x) for name, head in task.items()} for task in self.task_heads]


class FastBEV(nn.Module):
    """Single-frame FastBEV baseline."""
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
        self.out_channels = out_channels

        backbone = resnet50(weights=None)
        self.img_backbone = backbone
        self.img_neck = CustomFPN(in_channels=[1024, 2048], out_channels=in_channels)
        self.img_view_transformer = FastrayTransformer(
            in_channels=in_channels,
            out_channels=bev_channels,
            image_size=image_size,
            feature_size=feature_size,
        )
        self.img_bev_encoder_backbone = CustomResNetBEV(
            numC_input=bev_channels,
            num_channels=[bev_channels * 2, bev_channels * 4, bev_channels * 8],
        )
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=bev_channels * 8 + bev_channels * 2,
            out_channels=out_channels,
        )
        self.pts_bbox_head = CenterHead(in_channels=out_channels, num_classes=num_classes)

    def extract_img_feat(self, img):
        x = self.img_backbone.conv1(img)
        x = self.img_backbone.bn1(x)
        x = self.img_backbone.relu(x)
        x = self.img_backbone.maxpool(x)
        x1 = self.img_backbone.layer1(x)
        x2 = self.img_backbone.layer2(x1)
        x3 = self.img_backbone.layer3(x2)
        x4 = self.img_backbone.layer4(x3)
        return self.img_neck([x3, x4])

    def encode(self, img, cam2ego, cam_intrinsics, img_aug_matrix=None):
        """Image → BEV features. Returns (bev_feat [B, C, H, W], depth [B, H, W])."""
        img_feats = self.extract_img_feat(img)
        bev_feat, depth = self.img_view_transformer(img, img_feats, cam2ego, cam_intrinsics, img_aug_matrix)
        bev_feats = self.img_bev_encoder_backbone(bev_feat)
        bev_feat = self.img_bev_encoder_neck(bev_feats)
        return bev_feat, depth

    def forward(self, img, cam2ego, cam_intrinsics, img_aug_matrix=None):
        bev_feat, depth = self.encode(img, cam2ego, cam_intrinsics, img_aug_matrix)
        preds = self.pts_bbox_head(bev_feat)
        return {'predictions': preds, 'bev_feat': bev_feat, 'depth': depth}


class FastBEV4D(FastBEV):
    """
    FastBEV with BEVDet4D-style temporal fusion (Option A: concat + 1×1 conv).

    Usage:
        # First frame (no history)
        out = model(img, cam2ego, intrinsics)

        # Subsequent frames
        out = model(img, cam2ego, intrinsics,
                    bev_feat_prev=prev_out['bev_feat'],
                    se2=se2_tensor)   # [B, 3] — (dx, dy, dyaw) in grid units
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_fusion = BEVTemporalFusionConcat(feat_channels=self.out_channels)

    def forward(
        self,
        img,
        cam2ego,
        cam_intrinsics,
        img_aug_matrix=None,
        bev_feat_prev: Optional[torch.Tensor] = None,
        se2: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            img:           [B, 3, H, W]   monocam image (ImageNet-normalised)
            cam2ego:       [B, 4, 4]
            cam_intrinsics:[B, 3, 3]
            img_aug_matrix:[B, 3, 3]  or None
            bev_feat_prev: [B, C, H, W]  BEV features from t-1, or None for first frame
            se2:           [B, 3]  (dx, dy, dyaw) in grid units, or None for first frame

        Returns:
            dict with 'predictions', 'bev_feat', 'depth'
        """
        bev_feat_enc, depth = self.encode(img, cam2ego, cam_intrinsics, img_aug_matrix)

        if bev_feat_prev is not None and se2 is not None:
            bev_feat = self.temporal_fusion(bev_feat_enc, bev_feat_prev, se2)
        else:
            bev_feat = bev_feat_enc

        preds = self.pts_bbox_head(bev_feat)
        return {
            'predictions': preds,
            'bev_feat':    bev_feat,      # post-fusion (fed to head)
            'bev_feat_enc': bev_feat_enc, # pre-fusion  (cache as prev for next frame)
            'depth':       depth,
        }


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load pretrained FastBEV checkpoint with automatic key remapping."""
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
        if ckpt_key in model_dict and model_dict[ckpt_key].shape == ckpt_val.shape:
            mapped_dict[ckpt_key] = ckpt_val
        else:
            unmatched.append(ckpt_key)

    model.load_state_dict(mapped_dict, strict=False)
    print(f"  Loaded {len(mapped_dict)}/{len(state_dict)} checkpoint keys")
    if unmatched:
        print(f"  Unmatched: {len(unmatched)} keys (temporal_fusion weights are new — expected)")
    return model
