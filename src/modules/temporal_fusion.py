"""
BEV temporal fusion module 
Includes spatial warping, feature fusion, and optional recurrent memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BEVWarp(nn.Module):
    """
    Spatially warp BEV features from previous frame to current frame
    using ego-motion (SE(2) transform).
    """

    def __init__(self, grid_size_m: float = 51.2):
        """
        Args:
            grid_size_m: Total BEV extent (e.g., [-51.2, 51.2]m).
        """
        super().__init__()
        self.grid_size_m = grid_size_m

    def forward(
        self,
        bev_feat_prev: torch.Tensor,
        se2_transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp previous BEV features to current frame coordinates.

        Args:
            bev_feat_prev: [B, C, H, W] BEV feature map from t-1
            se2_transform: [B, 3] with [dx, dy, dyaw] in grid units

        Returns:
            warped: [B, C, H, W] features transformed to t's coordinate frame
        """
        B, C, H, W = bev_feat_prev.shape

        # Extract transform components
        dx = se2_transform[:, 0]  # [B]
        dy = se2_transform[:, 1]  # [B]
        dyaw = se2_transform[:, 2]  # [B]

        # Build rotation matrix
        cos_yaw = torch.cos(dyaw)  # [B]
        sin_yaw = torch.sin(dyaw)  # [B]

        # Normalize translation to [-1, 1] for grid_sample
        # Grid coordinates range from -1 to 1 across the BEV extent
        tx_norm = dx / (W / 2.0)  # [B]
        ty_norm = dy / (H / 2.0)  # [B]

        # Build 2×3 affine matrix for each sample in batch
        # [cos, -sin, tx]
        # [sin,  cos, ty]
        affine_matrices = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, tx_norm], dim=1),
            torch.stack([sin_yaw,  cos_yaw, ty_norm], dim=1),
        ], dim=1)  # [B, 2, 3]

        # Generate sampling grid
        grid = F.affine_grid(affine_matrices, size=(B, C, H, W))

        # Warp using bilinear interpolation, pad with zeros for out-of-bounds
        warped = F.grid_sample(
            bev_feat_prev,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        return warped


class BEVTemporalFusionConcat(nn.Module):
    """
    Simple Option A: Concatenate current and warped previous features,
    then squeeze with a 1×1 convolution.
    Fast and easy to debug.
    """

    def __init__(self, feat_channels: int = 256):
        """
        Args:
            feat_channels: Number of channels in BEV feature maps.
        """
        super().__init__()
        self.warp = BEVWarp()
        self.fusion_conv = nn.Conv2d(
            feat_channels * 2,
            feat_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        bev_feat_curr: torch.Tensor,
        bev_feat_prev: torch.Tensor,
        se2_transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse current and previous BEV features.

        Args:
            bev_feat_curr: [B, C, H, W] Current frame BEV features
            bev_feat_prev: [B, C, H, W] Previous frame BEV features
            se2_transform: [B, 3] with [dx, dy, dyaw] ego-motion

        Returns:
            fused: [B, C, H, W] Temporally fused features
        """
        # Warp previous features to current frame
        warped_prev = self.warp(bev_feat_prev, se2_transform)

        # Concatenate along channel dimension
        concatenated = torch.cat([bev_feat_curr, warped_prev], dim=1)  # [B, 2C, H, W]

        # Squeeze back to original channel count
        fused = self.fusion_conv(concatenated)  # [B, C, H, W]

        return fused


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for recurrent processing of BEV feature maps.
    Maintains hidden state across time steps.
    """

    def __init__(self, feat_channels: int = 256):
        """
        Args:
            feat_channels: Number of channels in feature maps.
        """
        super().__init__()
        self.feat_channels = feat_channels

        # Gates: reset, update
        self.conv_gates = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Candidate hidden state
        self.conv_candidate = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single GRU step on BEV feature maps.

        Args:
            x: [B, C, H, W] Input features (e.g., concatenated current + warped prev)
            h_prev: [B, C, H, W] Previous hidden state

        Returns:
            h_new: [B, C, H, W] Updated hidden state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # [B, 2C, H, W]

        # Compute reset and update gates
        gates = self.conv_gates(combined)
        reset_gate, update_gate = gates.split(self.feat_channels, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # Compute candidate hidden state
        combined_candidate = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = self.conv_candidate(combined_candidate)

        # Update hidden state
        h_new = (1 - update_gate) * candidate + update_gate * h_prev

        return h_new


class BEVTemporalFusionGRU(nn.Module):
    """
    Option B: Recurrent fusion with ConvGRU for richer temporal memory.
    Better for handling long occlusions but adds ~2M params and training complexity.
    """

    def __init__(self, feat_channels: int = 256):
        """
        Args:
            feat_channels: Number of channels in BEV feature maps.
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.warp = BEVWarp()
        self.gru_cell = ConvGRUCell(feat_channels)

    def forward(
        self,
        bev_feat_curr: torch.Tensor,
        bev_feat_prev: torch.Tensor,
        se2_transform: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse current and previous features with recurrent memory.

        Args:
            bev_feat_curr: [B, C, H, W] Current frame BEV features
            bev_feat_prev: [B, C, H, W] Previous frame BEV features
            se2_transform: [B, 3] ego-motion
            hidden_state: [B, C, H, W] or None. If None, initialize as zeros.

        Returns:
            output: [B, C, H, W] Fused features
            hidden_state: [B, C, H, W] Updated hidden state for next time step
        """
        B, C, H, W = bev_feat_curr.shape

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(B, C, H, W, dtype=bev_feat_curr.dtype, device=bev_feat_curr.device)

        # Warp previous features
        warped_prev = self.warp(bev_feat_prev, se2_transform)

        # Concatenate current and warped prev as input to GRU
        gru_input = torch.cat([bev_feat_curr, warped_prev], dim=1)  # [B, 2C, H, W]

        # GRU step
        hidden_state = self.gru_cell(gru_input, hidden_state)

        return hidden_state, hidden_state
