"""
Geometric hypothesis test for BEV cross-modal alignment.

Tests whether the dense displacement field learned by DisplacementHead
(H0: spatially-varying non-rigid warp) is well-approximated by a global
Sim(2) / SE(2) transform (H1: rigid Lie-group alignment).

H1_geo : The dense field is approximately explainable by Sim(2) with
         small residual  →  high R²
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

    grid = create_meshgrid(H, W, normalized_coordinates=True, device=device)
    src_pts = grid.reshape(-1, 2).double()   # (H*W, 2)

    # Target points = source + displacement (delta already normalised)
    norm_delta = torch.stack([
        delta[0, 0],   # x offsets
        delta[0, 1],   # y offsets
    ], dim=-1).reshape(-1, 2).double()

    dst_pts = src_pts + norm_delta   # (H*W, 2)

    # Build Sim(2) linear system  A·[a, b, tx, ty]ᵀ = dst
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

    # Residuals and R²
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


def visualize_geometry_comparison(
    delta: torch.Tensor,
    params: Dict,
    sample_idx: int,
    save_path: str = None,
):
    """
    Quiver plot comparing H0 (dense flow, red) vs H1 (Sim(2) fit, blue).

    A high overlap between the two arrow fields supports H1 (rigid);
    large divergence supports H0 (non-rigid).
    """
    device = delta.device
    dense_flow = delta[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    H, W = dense_flow.shape[:2]

    # Re-derive the Sim(2) flow field from fitted parameters
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
    # Convert normalised units back to pixel units for the quiver scale
    rigid_flow[..., 0] *= W / 2
    rigid_flow[..., 1] *= H / 2

    dense_flow_px = dense_flow.copy()
    dense_flow_px[..., 0] *= W / 2
    dense_flow_px[..., 1] *= H / 2

    step = 8
    X_grid, Y_grid = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))

    plt.figure(figsize=(12, 12))
    plt.quiver(
        X_grid, Y_grid,
        dense_flow_px[::step, ::step, 0],
        dense_flow_px[::step, ::step, 1],
        color='red', alpha=0.4, label='H0: Dense Flow (Learned)',
        angles='xy', scale_units='xy', scale=1.0,
    )
    plt.quiver(
        X_grid, Y_grid,
        rigid_flow[::step, ::step, 0],
        rigid_flow[::step, ::step, 1],
        color='blue', alpha=0.7, label='H1: Rigid Sim(2) Fit',
        angles='xy', scale_units='xy', scale=1.0,
    )
    plt.title(
        f"Hypothesis Test (Sample {sample_idx}): H0 (Dense) vs H1 (Rigid)\n"
        f"R²={params['r2']:.4f}  |  Geodesic Dist={params['geodesic_dist']:.4f}\n"
        f"Sim(2): Rot={params['theta_deg']:.2f}°  "
        f"Trans=[{params['tx']:.3f}, {params['ty']:.3f}]  "
        f"Scale={params['scale']:.4f}"
    )
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved comparison plot → {save_path}")
    plt.close()


def visualize_residual_heatmap(
    residuals: torch.Tensor,
    sample_idx: int,
    save_path: str = None,
):
    """
    Heatmap of per-pixel non-rigid residual magnitude.

    Dark regions → Sim(2) explains the displacement well  (supports H1).
    Bright regions → large local warp unaccounted for by Sim(2)  (supports H0).

    Parameters
    ----------
    residuals : [H, W, 2] float tensor of non-rigid error from run_geometric_test.
    """
    mag = torch.norm(residuals, dim=-1).cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(mag, cmap='magma')
    plt.colorbar(label='Residual Magnitude (Normalised Units)')
    plt.title(f'Residual Heatmap (Sample {sample_idx}): Where Sim(2) (H1) Fails')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved residual heatmap → {save_path}")
    plt.close()

def visualize_bev_alignment():
    pass