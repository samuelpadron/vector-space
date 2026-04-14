"""
Geometric hypothesis test for BEV cross-modal alignment.

Tests whether the dense displacement field learned by DisplacementHead
(H0: spatially-varying non-rigid warp) is well-approximated by a global
Sim(2) / SE(2) transform (H1: rigid Lie-group alignment).

H1_geo : The dense field is approximately explainable by Sim(2) with
         small residual  →  high R², small geodesic distance.
H0_geo : The residual after Sim(2) fitting remains large and structured
         →  low R², large spatially-structured residual heatmap.

Key outputs per sample
----------------------
R²              — fraction of displacement variance explained by Sim(2).
                  R² ≈ 1 supports H1; R² ≪ 1 supports H0.
geodesic_dist   — norm of the SE(2) Lie algebra element (shortest path
                  on the manifold); represents overall misalignment magnitude.
theta_deg       — recovered yaw rotation in degrees.
tx, ty          — recovered translation in normalised [-1,1] BEV units.
scale           — recovered isotropic scale factor (Sim(2) extension).
residuals       — [H, W, 2] per-pixel non-rigid remainder after Sim(2) fit.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import kornia.geometry.liegroup as KLieGroup
from kornia.utils import create_meshgrid


def optical_flow_sim2(
    cam_bev: torch.Tensor,
    lidar_bev: torch.Tensor,
) -> Tuple[Dict, float, torch.Tensor]:
    """
    Classical optical flow

    Collapses cam_bev and lidar_bev (both mean across all channels) to
    single-channel images, runs Farneback optical flow between
    them, then feeds the resulting dense flow field into run_geometric_test.

    This provides a parameter-free baseline: if the Farneback flow produces
    similar R² and Sim(2) parameters to the learned DisplacementHead, the
    result is robust to the choice of flow estimation method.

    Parameters
    ----------
    cam_bev   : [1, C, H, W]  camera BEV features
    lidar_bev : [1, 4, H, W]  hand-crafted LiDAR BEV (ch0 = occupancy)

    Returns
    -------
    params, r2, residuals — same format as run_geometric_test
    """
    import cv2

    H, W = cam_bev.shape[2], cam_bev.shape[3]

    # Collapse to single channel
    cam_gray   = cam_bev[0].mean(dim=0).detach().cpu().numpy()    # [H, W]
    lidar_gray = lidar_bev[0].mean(dim=0).detach().cpu().numpy()  # [H, W] mean across channels

    # Normalise to uint8 [0, 255] — required by cv2.calcOpticalFlowFarneback
    def to_uint8(x: np.ndarray) -> np.ndarray:
        x = x - x.min()
        denom = x.max()
        if denom > 1e-8:
            x = x / denom
        return (x * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        to_uint8(cam_gray),
        to_uint8(lidar_gray),
        None,
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 15,
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0,
    )
    # flow: [H, W, 2] in pixel units — convert to normalised [-1, 1] units
    flow_norm = flow.copy()
    flow_norm[..., 0] /= (W / 2)   # x channel
    flow_norm[..., 1] /= (H / 2)   # y channel

    # Pack into [1, 2, H, W] tensor for run_geometric_test
    delta_flow = torch.from_numpy(
        flow_norm.transpose(2, 0, 1)[np.newaxis]
    ).float().to(cam_bev.device)

    return run_geometric_test(delta_flow, H, W)


def run_geometric_test(
    delta: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[Dict, float, torch.Tensor]:
    """
    Fit a Sim(2) model to a dense displacement field via least squares,
    then compute the geodesic distance on the SE(2) manifold and R².

    Parameters
    ----------
    delta : [1, 2, H, W]  normalised displacement field (output of
            DisplacementHead — values in [-1, 1]).
    H, W  : spatial dimensions of the BEV grid.

    Returns
    -------
    params       : dict with keys tx, ty, theta_deg, scale, r2,
                   geodesic_dist.
    r2_score     : float
    residuals    : FloatTensor [H, W, 2]  per-pixel non-rigid error.

    Algorithm
    ---------
    Sim(2) maps source point (x,y) → (ax−by+tx, bx+ay+ty) where
    a = s·cos θ, b = s·sin θ.  This is linear in [a, b, tx, ty],
    so we solve a 2N×4 least-squares system with torch.linalg.lstsq.
    """
    device = delta.device

    # Source grid in normalised [-1, 1] coordinates
    grid = create_meshgrid(H, W, normalized_coordinates=True, device=device)
    src_pts = grid.reshape(-1, 2).double()   # (H*W, 2)

    # Target points = source + displacement (delta already normalised)
    norm_delta = torch.stack([
        delta[0, 0],   # x offsets
        delta[0, 1],   # y offsets
    ], dim=-1).reshape(-1, 2).double()

    dst_pts = src_pts + norm_delta   # (H*W, 2)

    # Build Sim(2) linear system
    #    Row 2i  : [ x, -y, 1, 0 ]
    #    Row 2i+1: [ y,  x, 0, 1 ]
    x, y = src_pts[:, 0], src_pts[:, 1]
    N = x.shape[0]

    A = torch.zeros(2 * N, 4, device=device, dtype=torch.float64)
    A[0::2, 0],  A[0::2, 1],  A[0::2, 2]  = x,  -y, 1.0
    A[1::2, 0],  A[1::2, 1],  A[1::2, 3]  = y,   x, 1.0

    B_vec = dst_pts.reshape(-1, 1)
    sol = torch.linalg.lstsq(A, B_vec).solution.squeeze()
    a, b, tx, ty = sol

    # Extract Sim(2) primitives via Kornia SO(2)
    scale = torch.sqrt(a ** 2 + b ** 2)

    rot_mat = torch.stack([
        torch.stack([a / scale, -b / scale]),
        torch.stack([b / scale,  a / scale]),
    ]).unsqueeze(0)   # [1, 2, 2]

    so2_element = KLieGroup.So2.from_matrix(rot_mat)
    theta_rad = so2_element.log().squeeze()   # scalar tensor

    # Geodesic distance on SE(2) manifold
    t_vec = torch.tensor([[tx, ty]], device=device, dtype=torch.float64)
    se2_element = KLieGroup.Se2(so2_element, t_vec)
    geodesic_dist = torch.norm(se2_element.log()).item()

    # Residuals and R2
    fitted_pts = (A @ sol).reshape(-1, 2)
    residuals = dst_pts - fitted_pts   # (H*W, 2) — non-rigid remainder

    ss_res = (residuals ** 2).sum()
    ss_tot = ((norm_delta - norm_delta.mean(dim=0)) ** 2).sum()
    r2_score = (1.0 - ss_res / (ss_tot + 1e-9)).item()

    params = {
        'tx':            tx.item(),
        'ty':            ty.item(),
        'theta_deg':     torch.rad2deg(theta_rad).item(),
        'scale':         scale.item(),
        'r2':            r2_score,
        'geodesic_dist': geodesic_dist,
    }

    return params, r2_score, residuals.reshape(H, W, 2).float()


def _make_quiver_data(delta: torch.Tensor, params: Dict, step: int = 8):
    """
    Compute dense-flow and Sim(2)-flow arrow arrays for quiver plots.

    Returns (X_grid, Y_grid, dense_px, rigid_px) all in pixel units,
    subsampled every `step` pixels.
    """
    device = delta.device
    dense_flow = delta[0].permute(1, 2, 0).cpu().numpy()   # [H, W, 2]
    H, W = dense_flow.shape[:2]

    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij',
    )
    grid = torch.stack([xs, ys], dim=-1).view(-1, 2)

    rad = np.deg2rad(params['theta_deg'])
    a = params['scale'] * np.cos(rad)
    b = params['scale'] * np.sin(rad)

    fit_x = a * grid[:, 0] - b * grid[:, 1] + params['tx']
    fit_y = b * grid[:, 0] + a * grid[:, 1] + params['ty']

    rigid_flow = torch.stack([fit_x - grid[:, 0], fit_y - grid[:, 1]], dim=-1)
    rigid_flow = rigid_flow.view(H, W, 2).cpu().numpy()
    rigid_flow[..., 0] *= W / 2
    rigid_flow[..., 1] *= H / 2

    dense_px = dense_flow.copy()
    dense_px[..., 0] *= W / 2
    dense_px[..., 1] *= H / 2

    X_grid, Y_grid = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    return X_grid, Y_grid, dense_px, rigid_flow, H, W


def _draw_bev_overlays(ax, detections, residual_mag=None, voxel_size=0.8,
                       grid_size=128, cmap='magma'):
    """
    Shared helper: draw ego marker, distance rings, detection boxes,
    and optionally a residual heatmap underlay onto a BEV axis.
    """
    class_names = ['car', 'truck', 'cnstr', 'bus', 'trailer',
                   'barrier', 'moto', 'bicycle', 'ped', 'cone']
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    cx = cy = grid_size // 2   # ego at centre

    # Optional residual underlay
    if residual_mag is not None:
        ax.imshow(residual_mag, cmap=cmap, origin='upper',
                  extent=[0, grid_size, grid_size, 0], alpha=0.85)
    else:
        ax.set_facecolor('#111111')

    # Distance rings
    for dist in [10, 20, 30, 40, 50]:
        r = dist / voxel_size
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                color='white', linestyle='--', alpha=0.35, linewidth=0.7))
        ax.text(cx + r + 0.5, cy, f'{dist}m', color='white',
                fontsize=6, alpha=0.6, va='center')

    # Direction labels
    off = 56
    for txt, dx, dy, ha, va in [
        ('FRONT', 0, -off, 'center', 'bottom'),
        ('BACK',  0,  off, 'center', 'top'),
        ('LEFT', -off, 0, 'right',  'center'),
        ('RIGHT', off, 0, 'left',   'center'),
    ]:
        ax.text(cx + dx, cy + dy, txt, color='lime' if txt == 'FRONT' else 'white',
                fontsize=7, ha=ha, va=va, fontweight='bold', alpha=0.8)

    # Ego vehicle rectangle (pointing up = forward)
    ego_l = 4.5 / voxel_size
    ego_w = 2.0 / voxel_size
    ego_rect = plt.Polygon([
        [cx - ego_w/2, cy - ego_l/2],
        [cx + ego_w/2, cy - ego_l/2],
        [cx + ego_w/2, cy + ego_l/2],
        [cx - ego_w/2, cy + ego_l/2],
    ], closed=True, facecolor='cyan', edgecolor='white', linewidth=1.5,
       alpha=0.8, zorder=10)
    ax.add_patch(ego_rect)
    ax.arrow(cx, cy - ego_l/2 - 1, 0, -4, head_width=1.5, head_length=1,
             fc='lime', ec='white', linewidth=1, zorder=11)

    # Detection boxes — coordinate frame: ego BEV with FRONT=UP
    for det in detections:
        # nuScenes: x=forward, y=left → BEV pixel: px = cx - y/res, py = cy - x/res
        px = cx - det['y'] / voxel_size
        py = cy - det['x'] / voxel_size

        w_px = det['w'] / voxel_size
        l_px = det['l'] / voxel_size
        yaw  = -det['yaw']   # flip sign: nuScenes CCW, image CW

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-w_px/2,  l_px/2],
                             [ w_px/2,  l_px/2],
                             [ w_px/2, -l_px/2],
                             [-w_px/2, -l_px/2],
                             [-w_px/2,  l_px/2]])
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = corners @ rot.T
        corners[:, 0] += px
        corners[:, 1] += py

        color = class_colors[det['class']]
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.5, zorder=9)
        ax.text(px, py - l_px/2 - 1,
                class_names[det['class']], fontsize=5.5,
                ha='center', va='bottom', color='white', zorder=10)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)   # origin upper-left, FRONT = up
    ax.set_aspect('equal')


def visualize_sample(
    sample_idx: int,
    images: torch.Tensor,
    cam_bev: torch.Tensor,
    lidar_bev: torch.Tensor,
    preds,
    delta: torch.Tensor,
    params: Dict,
    residuals: torch.Tensor,
    detections: list,
    gt_boxes: list = None,
    save_path: str = None,
):
    """
    Combined 3-row figure for one sample:

    Row 1 — 6 camera images
    Row 2 — BEV features | detection heatmap | detections BEV | H0 vs H1 quiver
    Row 3 — LiDAR occ + predicted dets | LiDAR occ + GT boxes | residual heatmap | residual overlay

    The GT overlay in row 3 is the LiDAR BEV validation panel: it directly shows
    whether occupied cells correspond to ground truth annotated objects, validating
    that the hand-crafted 4-channel BEV faithfully represents scene geometry.

    Parameters
    ----------
    images    : [6, 3, H, W]  ImageNet-normalised camera images
    cam_bev   : [1, C, H, W]  camera BEV feature map
    lidar_bev : [1, 4, H, W]  hand-crafted LiDAR BEV (ch0 = occupancy)
    preds     : raw CenterHead output for heatmap display
    delta     : [1, 2, H, W]  learned displacement field
    params    : dict from run_geometric_test
    residuals : [H, W, 2]     per-pixel non-rigid residual
    detections: list of decoded prediction dicts (ego frame)
    gt_boxes  : list of ground truth box dicts (ego frame) from load_gt_boxes()
    """
    VOXEL_SIZE  = 0.8
    GRID_SIZE   = cam_bev.shape[-1]   # 128

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']

    # Pre-compute reused quantities
    residual_mag = torch.norm(residuals, dim=-1).cpu().numpy()   # [H, W]
    X_grid, Y_grid, dense_px, rigid_px, H_bev, W_bev = _make_quiver_data(delta, params)
    step = 8

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(28, 16))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(
        3, 8,
        height_ratios=[1, 1.4, 1.4],
        hspace=0.32, wspace=0.15,
        left=0.04, right=0.97, top=0.93, bottom=0.04,
    )

    # ── Row 1: Cameras ────────────────────────────────────────────────────
    for i, name in enumerate(cam_names):
        ax = fig.add_subplot(gs[0, i])
        img = (images[i].cpu() * std + mean).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(name, fontsize=8, color='white', pad=3)
        ax.axis('off')

    # ── Row 2, col 0-1: BEV features ─────────────────────────────────────
    ax_bev = fig.add_subplot(gs[1, 0:2])
    bev_vis = cam_bev[0].mean(dim=0).detach().cpu().numpy()
    bev_vis = np.rot90(bev_vis, k=1)
    ax_bev.imshow(bev_vis, cmap='viridis', origin='upper')
    ax_bev.set_title('Camera BEV Features (mean)', color='white', fontsize=9)
    ax_bev.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_bev.set_ylabel('← Back  |  Front →', color='white', fontsize=7)
    ax_bev.tick_params(colors='white', labelsize=6)
    for spine in ax_bev.spines.values():
        spine.set_edgecolor('#444')

    # ── Row 2, col 2-3: Detection heatmap ────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 2:4])
    heatmap = preds[0]['heatmap'][0].sigmoid().max(dim=0)[0].detach().cpu().numpy()
    heatmap = np.rot90(heatmap, k=1)
    ax_heat.imshow(heatmap, cmap='hot', origin='upper')
    ax_heat.set_title('CenterHead Detection Heatmap', color='white', fontsize=9)
    ax_heat.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_heat.tick_params(colors='white', labelsize=6)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor('#444')

    # ── Row 2, col 4-7: Quiver H0 vs H1 ─────────────────────────────────
    ax_quiver = fig.add_subplot(gs[1, 4:8])
    ax_quiver.set_facecolor('#0d0d1a')
    ax_quiver.quiver(
        X_grid, Y_grid,
        dense_px[::step, ::step, 0], dense_px[::step, ::step, 1],
        color='#ff6b6b', alpha=0.6, label='H0: Dense Flow',
        angles='xy', scale_units='xy', scale=1.0,
    )
    ax_quiver.quiver(
        X_grid, Y_grid,
        rigid_px[::step, ::step, 0], rigid_px[::step, ::step, 1],
        color='#4ecdc4', alpha=0.85, label='H1: Sim(2) Fit',
        angles='xy', scale_units='xy', scale=1.0,
    )
    ax_quiver.set_xlim(0, W_bev)
    ax_quiver.set_ylim(H_bev, 0)
    ax_quiver.set_title(
        f'H0 vs H1  —  R²={params["r2"]:.3f}  |  Geodesic={params["geodesic_dist"]:.3f}\n'
        f'Rot={params["theta_deg"]:.2f}°  Trans=[{params["tx"]:.3f}, {params["ty"]:.3f}]'
        f'  Scale={params["scale"]:.3f}',
        color='white', fontsize=8,
    )
    ax_quiver.legend(fontsize=7, loc='upper right',
                     facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)
    ax_quiver.tick_params(colors='white', labelsize=6)
    for spine in ax_quiver.spines.values():
        spine.set_edgecolor('#444')

    # ── Row 3, col 0-1: LiDAR occupancy + predicted detections ───────────
    ax_pred = fig.add_subplot(gs[2, 0:2])
    lidar_occ = lidar_bev[0, 0].cpu().numpy()
    _draw_bev_overlays(ax_pred, detections, residual_mag=None,
                       voxel_size=VOXEL_SIZE, grid_size=GRID_SIZE)
    ax_pred.imshow(lidar_occ, cmap='Blues', origin='upper',
                   extent=[0, GRID_SIZE, GRID_SIZE, 0], alpha=0.9)
    ax_pred.set_title('LiDAR Occupancy + Predicted Boxes', color='white', fontsize=9)
    ax_pred.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_pred.set_ylabel('← Back  |  Front →', color='white', fontsize=7)
    ax_pred.tick_params(colors='white', labelsize=6)
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('#444')

    # ── Row 3, col 2-3: LiDAR occupancy + GT boxes (validation panel) ────
    ax_gt = fig.add_subplot(gs[2, 2:4])
    _draw_bev_overlays(ax_gt, gt_boxes if gt_boxes is not None else [],
                       residual_mag=None, voxel_size=VOXEL_SIZE, grid_size=GRID_SIZE)
    ax_gt.imshow(lidar_occ, cmap='Blues', origin='upper',
                 extent=[0, GRID_SIZE, GRID_SIZE, 0], alpha=0.9)
    ax_gt.set_title('LiDAR Occupancy + Ground Truth Boxes\n'
                    '(validates: occupied cells ↔ annotated objects)',
                    color='lime', fontsize=9)
    ax_gt.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_gt.tick_params(colors='white', labelsize=6)
    for spine in ax_gt.spines.values():
        spine.set_edgecolor('lime')   # highlight as validation panel

    # ── Row 3, col 4-5: Residual heatmap ─────────────────────────────────
    ax_res = fig.add_subplot(gs[2, 4:6])
    im = ax_res.imshow(residual_mag, cmap='magma', origin='upper')
    cb = fig.colorbar(im, ax=ax_res, fraction=0.046, pad=0.04)
    cb.set_label('Residual Magnitude', color='white', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='white', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    ax_res.set_title('Non-Rigid Residual: Where Sim(2) Fails', color='white', fontsize=9)
    ax_res.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_res.tick_params(colors='white', labelsize=6)
    for spine in ax_res.spines.values():
        spine.set_edgecolor('#444')

    # ── Row 3, col 6-7: Residual + detections overlay ────────────────────
    ax_overlay = fig.add_subplot(gs[2, 6:8])
    _draw_bev_overlays(ax_overlay, detections, residual_mag=residual_mag,
                       voxel_size=VOXEL_SIZE, grid_size=GRID_SIZE, cmap='magma')
    ax_overlay.set_title('Residual + Detected Objects', color='white', fontsize=9)
    ax_overlay.set_xlabel('← Left  |  Right →', color='white', fontsize=7)
    ax_overlay.tick_params(colors='white', labelsize=6)
    for spine in ax_overlay.spines.values():
        spine.set_edgecolor('#444')

    fig.suptitle(
        f'BEV Alignment Hypothesis Test — Sample {sample_idx}',
        fontsize=13, color='white', fontweight='bold', y=0.97,
    )

    # Set all axis backgrounds dark
    for ax in fig.get_axes():
        ax.set_facecolor('#1a1a2e')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Saved combined visualisation → {save_path}")
    plt.close()