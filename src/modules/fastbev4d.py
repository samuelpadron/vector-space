"""
fastbev4d.py — FastBEV with BEVDet4D-style temporal fusion.
Pure PyTorch, no mmcv/mmdet dependencies.
Based on FastBEV++ paper: https://arxiv.org/abs/2512.08237
and BEVDet4d paper: https://arxiv.org/abs/2203.17054
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.temporal_fusion import BEVTemporalFusionConcat


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

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs[0]


class FastrayTransformer(nn.Module):
    """
    FastBEV ray-based image-to-BEV view transformer.

    Works for any number of cameras N >= 1.  For monocam, simply pass
    tensors with N=1 on the camera dimension.

    The depth head (depth_net) predicts D+C channels jointly:
      - first D channels → categorical depth distribution (softmax)
      - last  C channels → image features projected to BEV channel count
    Both are learned end-to-end via the detection loss — no external depth
    supervision needed.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        image_size: Tuple[int, int] = (256, 704),
        feature_size: Tuple[int, int] = (16, 44),
        grid_config: Dict = None,
        stride: int = 16,
        use_bevpool: bool = False,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.image_size   = image_size
        self.feature_size = feature_size
        self.stride       = stride
        self.use_bevpool  = use_bevpool

        if grid_config is None:
            grid_config = {
                'x':     [-51.2, 51.2, 0.8],
                'y':     [-51.2, 51.2, 0.8],
                'z':     [-2.5,  4.5,  1.0],
                'depth': [1.0,  60.0,  1.0],
            }
        self.grid_config = grid_config

        self.X = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])  # 128
        self.Y = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])  # 128
        self.Z = int((grid_config['z'][1] - grid_config['z'][0]) / grid_config['z'][2])  # 7
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])  # 59

        self.grid_lower_bound = torch.tensor(
            [grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]]
        )
        self.grid_interval = torch.tensor(
            [grid_config['x'][2], grid_config['y'][2], grid_config['z'][2]]
        )

        # Single conv predicts depth distribution + projected features together.
        # This is the original FastBEV design — cheaper and works just as well
        # as a two-stage approach for this task.
        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, kernel_size=1, padding=0)

        self.register_buffer('voxel_coords', self._create_voxel_coords())

    def _create_voxel_coords(self) -> torch.Tensor:
        x = torch.arange(self.X).view(-1, 1, 1).expand(-1, self.Y, self.Z).float()
        y = torch.arange(self.Y).view(1, -1, 1).expand(self.X, -1, self.Z).float()
        z = torch.arange(self.Z).view(1, 1, -1).expand(self.X, self.Y, -1).float()
        coords = torch.stack((x, y, z), dim=3)
        coords = coords * self.grid_interval + self.grid_lower_bound
        return coords.reshape(-1, 3)

    def forward(
        self,
        img_feats,            # [B, N, C, H, W]
        cam2ego,              # [B, N, 4, 4]
        cam_intrinsics,       # [B, N, 3, 3]
        img_aug_matrix=None,
    ):
        """
        Returns
        -------
        bev_feat  : [B, out_channels, Y, X]
        depth     : [B, N, H, W, D]   softmax depth distribution per camera
        """
        B, N, C, H, W = img_feats.shape

        x = img_feats.view(B * N, C, H, W)
        x = self.depth_net(x)
        x = x.view(B, N, self.D + self.out_channels, H, W)
        x = x.permute(0, 1, 3, 4, 2)   # [B, N, H, W, D+C]

        depth = x[..., :self.D].softmax(dim=-1)  # [B, N, H, W, D]
        feat  = x[..., self.D:]                   # [B, N, H, W, out_channels]

        if self.use_bevpool:
            bev_feat = self._image_to_bev(feat, depth, cam2ego, cam_intrinsics)
        else:
            bev_feat = self._project_and_sample(feat, depth, cam2ego, cam_intrinsics)
        return bev_feat, depth

    def _project_and_sample(self, feat, depth, cam2ego, cam_intrinsics) -> torch.Tensor:
        B, N, H, W, C = feat.shape
        device = feat.device

        bev_feat   = torch.zeros(B, self.X, self.Y, self.Z, C, device=device, dtype=feat.dtype)
        voxel_coords = self.voxel_coords.to(device)
        num_voxels   = voxel_coords.shape[0]

        for b in range(B):
            for n in range(N):
                c2e = cam2ego[b, n]          # [4, 4]
                K   = cam_intrinsics[b, n]   # [3, 3]
                e2c = torch.linalg.inv(c2e)

                # project voxel centres into camera frame
                voxel_homo = torch.cat(
                    [voxel_coords, torch.ones(num_voxels, 1, device=device)], dim=1
                )                                              # [N_vox, 4]
                cam_coords = (e2c @ voxel_homo.T).T[:, :3]   # [N_vox, 3]

                z       = cam_coords[:, 2]                    # metric depth
                valid_z = z > 0.5

                z_safe = z.clamp(min=0.1)
                # full projection via homogeneous coords
                cam_norm  = cam_coords / z_safe.unsqueeze(-1)
                cam_homo  = torch.cat(
                    [cam_norm[:, :2], torch.ones(num_voxels, 1, device=device)], dim=1
                )
                img_coords = (K @ cam_homo.T).T[:, :2]        # [N_vox, 2]
                feat_coords = img_coords / self.stride

                valid_u = (feat_coords[:, 0] >= 0) & (feat_coords[:, 0] < W)
                valid_v = (feat_coords[:, 1] >= 0) & (feat_coords[:, 1] < H)

                depth_bin = (
                    (z - self.grid_config['depth'][0]) / self.grid_config['depth'][2]
                ).long()
                valid_d = (depth_bin >= 0) & (depth_bin < self.D)

                valid     = valid_z & valid_u & valid_v & valid_d
                valid_idx = torch.where(valid)[0]
                if valid_idx.numel() == 0:
                    continue

                fx = feat_coords[valid_idx, 0]
                fy = feat_coords[valid_idx, 1]
                d  = depth_bin[valid_idx].clamp(0, self.D - 1)

                x0 = fx.floor().long().clamp(0, W - 2)
                y0 = fy.floor().long().clamp(0, H - 2)
                x1 = x0 + 1
                y1 = y0 + 1
                wx = (fx - x0.float()).clamp(0.0, 1.0).unsqueeze(-1)  # [M, 1]
                wy = (fy - y0.float()).clamp(0.0, 1.0).unsqueeze(-1)  # [M, 1]

                sampled_feat = (
                    (1 - wy) * ((1 - wx) * feat[b, n, y0, x0, :] + wx * feat[b, n, y0, x1, :]) +
                         wy  * ((1 - wx) * feat[b, n, y1, x0, :] + wx * feat[b, n, y1, x1, :])
                )
                sampled_depth = (
                    (1 - wy[:, 0]) * ((1 - wx[:, 0]) * depth[b, n, y0, x0, d] + wx[:, 0] * depth[b, n, y0, x1, d]) +
                         wy[:, 0]  * ((1 - wx[:, 0]) * depth[b, n, y1, x0, d] + wx[:, 0] * depth[b, n, y1, x1, d])
                )
                weighted_feat = sampled_feat * sampled_depth.unsqueeze(-1)

                vx       = valid_idx // (self.Y * self.Z)
                vy       = (valid_idx % (self.Y * self.Z)) // self.Z
                vz       = valid_idx % self.Z
                flat_idx = vx * self.Y * self.Z + vy * self.Z + vz

                bev_flat = bev_feat[b].view(-1, C)
                bev_flat.scatter_add_(
                    0, flat_idx.unsqueeze(-1).expand(-1, C), weighted_feat.to(bev_flat.dtype)
                )

        bev_feat = bev_feat.sum(dim=3)           # collapse Z → [B, X, Y, C]
        bev_feat = bev_feat.permute(0, 3, 2, 1)  # [B, C, Y, X]
        return bev_feat

    def _image_to_bev(self, feat, depth, cam2ego, cam_intrinsics) -> torch.Tensor:
        """
        BEVPoolv2-equivalent accumulation: image→BEV direction.

        Iterates over all (feature pixel, depth bin) pairs for each camera and
        scatters feat[h,w] * depth_prob[h,w,d] into the corresponding BEV voxel.
        This matches the original BEVPoolv2 CUDA kernel's accumulation pattern and
        must be used when evaluating pretrained FastBEV++ checkpoints.
        """
        B, N, H, W, C = feat.shape
        device = feat.device

        # Feature-pixel centres in image space: pixel (w_f, h_f) → (u, v)
        h_idx = torch.arange(H, device=device, dtype=torch.float32)
        w_idx = torch.arange(W, device=device, dtype=torch.float32)
        img_v, img_u = torch.meshgrid(
            h_idx * self.stride + self.stride * 0.5,
            w_idx * self.stride + self.stride * 0.5,
            indexing='ij',
        )  # [H, W]

        # Depth bin lower bounds — matches torch.arange(*grid_config['depth'])
        depth_vals = (
            self.grid_config['depth'][0]
            + torch.arange(self.D, device=device, dtype=torch.float32)
            * self.grid_config['depth'][2]
        )  # [D]

        # Grid origin derived from the (checkpoint-loaded) voxel_coords buffer
        x0 = self.voxel_coords[0, 0].item()
        y0 = self.voxel_coords[0, 1].item()
        z0 = self.voxel_coords[0, 2].item()
        dx = self.grid_config['x'][2]
        dy = self.grid_config['y'][2]
        dz = self.grid_config['z'][2]

        # Homogeneous image coords for all H*W feature pixels [H*W, 3]
        ones_hw = torch.ones(H * W, device=device, dtype=torch.float32)
        uv1 = torch.stack(
            [img_u.reshape(-1), img_v.reshape(-1), ones_hw], dim=1
        )  # [H*W, 3]

        bev_feat = torch.zeros(B, self.X, self.Y, self.Z, C, device=device, dtype=feat.dtype)

        for b in range(B):
            for n in range(N):
                K_inv = torch.linalg.inv(cam_intrinsics[b, n])  # [3, 3]
                c2e   = cam2ego[b, n]                            # [4, 4]

                # Normalised camera-space ray directions [H*W, 3]
                # K_inv @ [u, v, 1]^T = [(u-cx)/fx, (v-cy)/fy, 1]
                cam_dirs = (K_inv @ uv1.T).T  # [H*W, 3]

                # 3D points for every (pixel, depth_bin) — [H*W, D, 3]
                cam_pts = cam_dirs.unsqueeze(1) * depth_vals.view(1, -1, 1)

                # Transform to ego frame — flatten first for batch matmul
                HWD = H * W * self.D
                cam_homo = torch.cat(
                    [cam_pts.reshape(HWD, 3),
                     torch.ones(HWD, 1, device=device, dtype=cam_pts.dtype)],
                    dim=1,
                )  # [HWD, 4]
                ego_pts = (c2e @ cam_homo.T).T[:, :3]  # [HWD, 3]

                # BEV voxel indices
                vx = ((ego_pts[:, 0] - x0) / dx).long()
                vy = ((ego_pts[:, 1] - y0) / dy).long()
                vz = ((ego_pts[:, 2] - z0) / dz).long()

                valid = (
                    (vx >= 0) & (vx < self.X)
                    & (vy >= 0) & (vy < self.Y)
                    & (vz >= 0) & (vz < self.Z)
                )
                valid_idx = valid.nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue

                # Map flat index → (h_f, w_f, d)
                d_v  = valid_idx % self.D
                hw_v = valid_idx // self.D
                h_v  = hw_v // W
                w_v  = hw_v % W

                depth_w = depth[b, n, h_v, w_v, d_v]      # [M]
                feats_v = feat[b, n, h_v, w_v, :]          # [M, C]
                weighted = feats_v * depth_w.unsqueeze(-1)  # [M, C]

                flat_bev = (
                    vx[valid_idx] * self.Y * self.Z
                    + vy[valid_idx] * self.Z
                    + vz[valid_idx]
                )
                bev_feat[b].view(-1, C).scatter_add_(
                    0, flat_bev.unsqueeze(-1).expand(-1, C), weighted.to(bev_feat.dtype)
                )

        bev_feat = bev_feat.sum(dim=3)            # [B, X, Y, C]
        bev_feat = bev_feat.permute(0, 3, 2, 1)  # [B, C, Y, X]
        return bev_feat


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1     = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1       = nn.BatchNorm2d(out_channels)
        self.conv2     = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2       = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu      = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CustomResNetBEV(nn.Module):
    def __init__(self, numC_input=64, num_channels=[128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_ch = numC_input
        for out_ch in num_channels:
            ds = nn.Conv2d(curr_ch, out_ch, 3, stride=2, padding=1)
            self.layers.append(nn.Sequential(
                BasicBlock(curr_ch, out_ch, stride=2, downsample=ds),
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
    """FPN neck for BEV features.
    Uses feats[0] (128ch, high-res) and feats[2] (512ch, low-res),
    skipping feats[1] to match original FastBEV architecture.
    """
    def __init__(self, in_channels=640, out_channels=256, scale_factor=4, extra_upsample=2):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
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
        x  = torch.cat([feats[0], x1], dim=1)
        return self.up2(self.conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu else x


class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

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
        self.task_heads  = nn.ModuleList([
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
    Single-frame FastBEV baseline.

    All tensors use the N-camera convention even for monocam (N=1):
        imgs       : [B, 1, 3, H, W]
        cam2ego    : [B, 1, 4, 4]
        intrinsics : [B, 1, 3, 3]
    """

    def __init__(
        self,
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
        use_bevpool=False,
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.out_channels = out_channels

        self.img_backbone = resnet50(weights=None)
        self.img_neck     = CustomFPN(in_channels=[1024, 2048], out_channels=in_channels)

        self.img_view_transformer = FastrayTransformer(
            in_channels=in_channels,
            out_channels=bev_channels,
            image_size=image_size,
            feature_size=feature_size,
            use_bevpool=use_bevpool,
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

    def extract_img_feat(self, imgs):
        """
        Args: imgs [B, N, 3, H, W]
        Returns: [B, N, C_out, H_feat, W_feat]
        """
        B, N, C, H, W = imgs.shape
        x = imgs.view(B * N, C, H, W)

        x  = self.img_backbone.maxpool(
                 self.img_backbone.relu(
                     self.img_backbone.bn1(
                         self.img_backbone.conv1(x))))
        x1 = self.img_backbone.layer1(x)
        x2 = self.img_backbone.layer2(x1)
        x3 = self.img_backbone.layer3(x2)   # 1024ch
        x4 = self.img_backbone.layer4(x3)   # 2048ch

        feat = self.img_neck([x3, x4])
        _, C_out, H_out, W_out = feat.shape
        return feat.view(B, N, C_out, H_out, W_out)

    def encode(self, imgs, cam2ego, cam_intrinsics, img_aug_matrix=None):
        """
        Args:
            imgs        [B, N, 3, H, W]
            cam2ego     [B, N, 4, 4]
            cam_intrinsics [B, N, 3, 3]
        Returns:
            bev_feat    [B, out_channels, H_bev, W_bev]
            depth       [B, N, H_feat, W_feat, D]
        """
        img_feats = self.extract_img_feat(imgs)
        bev_feat, depth = self.img_view_transformer(
            img_feats, cam2ego, cam_intrinsics, img_aug_matrix
        )
        bev_feats = self.img_bev_encoder_backbone(bev_feat)
        bev_feat  = self.img_bev_encoder_neck(bev_feats)
        return bev_feat, depth

    def forward(self, imgs, cam2ego, cam_intrinsics, img_aug_matrix=None):
        bev_feat, depth = self.encode(imgs, cam2ego, cam_intrinsics, img_aug_matrix)
        preds = self.pts_bbox_head(bev_feat)
        return {'predictions': preds, 'bev_feat': bev_feat, 'depth': depth}


class FastBEV4D(FastBEV):
    """
    FastBEV with multi-frame temporal fusion.

    prev_frame controls how many frames are fused
    """

    def __init__(self, *args, num_prev_frames=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_prev_frames = num_prev_frames
        self.temporal_fusion = BEVTemporalFusionConcat(
            feat_channels=self.out_channels,
            num_prev_frames=self.num_prev_frames,
            dropout=0.3,
        )

    def forward(self, imgs, cam2ego, cam_intrinsics, img_aug_matrix=None,
                bev_feats_prev=None, se2_list=None):
        """
        Args:
            imgs, cam2ego, cam_intrinsics: current frame tensors
            bev_feats_prev: list of T tensors [B, C, H, W], one per offset in
                            prev_frame_offsets. None to skip fusion.
            se2_list: [B, T, 3] ego-motion from each past frame to current.

        Returns dict with keys: predictions, bev_feat, bev_feat_enc, depth.
        """
        img_feats = self.extract_img_feat(imgs)
        bev_feat_sparse, depth = self.img_view_transformer(
            img_feats, cam2ego, cam_intrinsics, img_aug_matrix
        )

        bev_feats = self.img_bev_encoder_backbone(bev_feat_sparse)
        bev_feat_enc = self.img_bev_encoder_neck(bev_feats)

        if bev_feats_prev is not None and se2_list is not None:
            bev_feat_fused = self.temporal_fusion(bev_feat_enc, bev_feats_prev, se2_list)
        else:
            bev_feat_fused = bev_feat_enc

        preds = self.pts_bbox_head(bev_feat_fused)

        return {
            'predictions':  preds,
            'bev_feat':     bev_feat_fused,
            'bev_feat_enc': bev_feat_enc,
            'depth':        depth,
        }


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load a pretrained FastBEV checkpoint.

    Keys that don't match (e.g. temporal_fusion, which is new) are skipped
    with a warning rather than raising an error.
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)   # handle both wrapped and raw dicts
    model_dict = model.state_dict()

    prefixes = (
        'img_backbone.',
        'img_neck.',
        'img_view_transformer.',
        'img_bev_encoder_backbone.',
        'img_bev_encoder_neck.',
        'pts_bbox_head.',
    )

    mapped, unmatched = {}, []
    for k, v in state_dict.items():
        if not any(k.startswith(p) for p in prefixes):
            unmatched.append(k)
            continue
        if k in model_dict and model_dict[k].shape == v.shape:
            mapped[k] = v
        else:
            unmatched.append(k)

    model.load_state_dict(mapped, strict=False)
    print(f"  Loaded  : {len(mapped)}/{len(state_dict)} keys")
    if unmatched:
        print(f"  Skipped : {len(unmatched)} keys "
              f"(temporal_fusion + any shape mismatches — expected)")
    return model