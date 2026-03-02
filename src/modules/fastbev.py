"""
FastBEV camera BEV model.
Pure PyTorch — no mmcv/mmdet dependencies.
Based on: https://arxiv.org/abs/2512.08237
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class CustomFPN(nn.Module):
    """Feature Pyramid Network neck matching original FastBEV architecture."""

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

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs[0]  # Return finest level


class FastrayTransformer(nn.Module):
    """FastBEV ray-based image-to-BEV view transformer."""

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

        self.X = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])   # 128
        self.Y = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])   # 128
        self.Z = int((grid_config['z'][1] - grid_config['z'][0]) / grid_config['z'][2])   # 7
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])  # 59

        self.grid_lower_bound = torch.tensor([grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]])
        self.grid_interval = torch.tensor([grid_config['x'][2], grid_config['y'][2], grid_config['z'][2]])

        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, kernel_size=1, padding=0)
        self.register_buffer('voxel_coords', self._create_voxel_coords())

    def _create_voxel_coords(self):
        x = torch.arange(self.X).view(-1, 1, 1).expand(-1, self.Y, self.Z).float()
        y = torch.arange(self.Y).view(1, -1, 1).expand(self.X, -1, self.Z).float()
        z = torch.arange(self.Z).view(1, 1, -1).expand(self.X, self.Y, -1).float()
        coords = torch.stack((x, y, z), dim=3)
        coords = coords * self.grid_interval + self.grid_lower_bound
        return coords.reshape(-1, 3)

    def forward(self, img_feats, cam2ego, cam_intrinsics, img_aug_matrix=None):
        B, N, C, H, W = img_feats.shape

        x = img_feats.view(B * N, C, H, W)
        x = self.depth_net(x)
        x = x.view(B, N, self.D + self.out_channels, H, W)
        x = x.permute(0, 1, 3, 4, 2)   # (B, N, H, W, D+C)

        depth = x[..., :self.D].softmax(dim=-1)
        feat = x[..., self.D:]

        bev_feat = self._project_and_sample(feat, depth, cam2ego, cam_intrinsics, img_aug_matrix)
        return bev_feat, depth

    def _project_and_sample(self, feat, depth, cam2ego, cam_intrinsics, img_aug_matrix):
        B, N, H, W, C = feat.shape
        device = feat.device

        bev_feat = torch.zeros(B, self.X, self.Y, self.Z, C, device=device, dtype=feat.dtype)
        voxel_coords = self.voxel_coords.to(device)
        num_voxels = voxel_coords.shape[0]

        for b in range(B):
            for n in range(N):
                c2e = cam2ego[b, n]
                K = cam_intrinsics[b, n]
                e2c = torch.inverse(c2e)

                voxel_homo = torch.cat([voxel_coords, torch.ones(num_voxels, 1, device=device)], dim=1)
                cam_coords = (e2c @ voxel_homo.T).T[:, :3]

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

                u = feat_coords[valid_idx, 0].long().clamp(0, W - 1)
                v = feat_coords[valid_idx, 1].long().clamp(0, H - 1)
                d = depth_bin[valid_idx].clamp(0, self.D - 1)

                sampled_feat = feat[b, n, v, u, :]
                sampled_depth = depth[b, n, v, u, d]
                weighted_feat = sampled_feat * sampled_depth.unsqueeze(-1)

                vx = valid_idx // (self.Y * self.Z)
                vy = (valid_idx % (self.Y * self.Z)) // self.Z
                vz = valid_idx % self.Z

                flat_idx = vx * self.Y * self.Z + vy * self.Z + vz
                bev_flat = bev_feat[b].view(-1, C)
                bev_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), weighted_feat)

        bev_feat = bev_feat.sum(dim=3)          # (B, X, Y, C)
        bev_feat = bev_feat.permute(0, 3, 2, 1) # (B, C, Y, X)
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

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class CustomResNetBEV(nn.Module):
    """BEV encoder backbone matching original FastBEV."""

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
    """FPN neck for BEV features matching original FastBEV.
    Note: uses feats[0] (128ch, high-res) and feats[2] (512ch, low-res),
    skipping feats[1] by design to match the original architecture.
    """

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
        x2 = feats[0]   # 128ch, higher res
        x1 = feats[2]   # 512ch, lower res (feats[1] skipped by design)
        x1 = self.up(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        return self.up2(x)


class ConvBNReLU(nn.Module):
    """Conv + BN + optional ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu else x


class ConvModule(nn.Module):
    """Conv + BN + ReLU matching mmcv ConvModule structure."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Sequential):
    """Per-output detection head matching CenterHead checkpoint structure:
    task_heads.0.<name>.0 = ConvModule, task_heads.0.<name>.1 = final Conv.
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
    """
    FastBEV: surround-view camera → BEV feature extractor.
    Outputs a [B, out_channels, H, W] BEV feature map alongside
    CenterPoint detection predictions.
    """

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = 64,
        out_channels: int = 256,
        num_classes: int = 10,
        image_size: Tuple[int, int] = (256, 704),
        feature_size: Tuple[int, int] = (16, 44),
    ):
        super().__init__()
        self.bev_channels = bev_channels

        self.img_backbone = resnet50(weights=None)
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
        # Input: 512ch (layer[2]) + 128ch (layer[0]) = 640ch
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=bev_channels * 8 + bev_channels * 2,
            out_channels=out_channels,
        )
        self.pts_bbox_head = CenterHead(in_channels=out_channels, num_classes=num_classes)

    def extract_img_feat(self, imgs):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)

        x = self.img_backbone.maxpool(
            self.img_backbone.relu(
                self.img_backbone.bn1(
                    self.img_backbone.conv1(imgs)
                )
            )
        )
        x1 = self.img_backbone.layer1(x)
        x2 = self.img_backbone.layer2(x1)
        x3 = self.img_backbone.layer3(x2)   # 1024ch
        x4 = self.img_backbone.layer4(x3)   # 2048ch

        feat = self.img_neck([x3, x4])
        _, C_out, H_out, W_out = feat.shape
        return feat.view(B, N, C_out, H_out, W_out)

    def forward(self, imgs, cam2ego, cam_intrinsics, img_aug_matrix=None):
        img_feats = self.extract_img_feat(imgs)
        bev_feat, depth = self.img_view_transformer(img_feats, cam2ego, cam_intrinsics, img_aug_matrix)
        bev_feats = self.img_bev_encoder_backbone(bev_feat)
        bev_feat = self.img_bev_encoder_neck(bev_feats)
        preds = self.pts_bbox_head(bev_feat)
        return {
            'predictions': preds,
            'bev_feat': bev_feat,
            'depth': depth,
        }
