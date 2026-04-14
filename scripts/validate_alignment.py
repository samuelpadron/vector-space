"""
validate_alignment.py — Synthetic Sim(2) recovery test for the alignment module.

Validates that the DisplacementHead + Sim(2) fitter pipeline correctly recovers
known ground-truth transforms.

Output
------
    Saved to validation_output/alignment_validation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV, HandcraftedLidarBEV, load_lidar_points, DisplacementHead, LidarProjector, apply_dense_warp
from data import load_sample, load_checkpoint
from hypothesis_test import run_geometric_test


NUSCENES_ROOT    = Path('./data/nuscenes')
CHECKPOINT_PATH  = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
OUTPUT_DIR       = Path('./validation_output')
GRID_CONF        = {'xbound': [-51.2, 51.2, 0.8], 'ybound': [-51.2, 51.2, 0.8]}
GRID_EXTENT      = 51.2   # metres — half-width of BEV
OPT_STEPS        = 1000
OPT_LR           = 1e-2
LIDAR_CHANNELS   = 4
CAMERA_CHANNELS  = 256

R2_THRESHOLD     = 0.90
THETA_THRESHOLD  = 1.0    # degrees
TRANS_THRESHOLD  = 0.01   # normalised units (~0.5m)

# Test cases — chosen to cover the range of realistic calibration errors
# tx_m, ty_m are in metres; theta_deg is rotation in degrees
TEST_CASES = [
    {'label': 'Identity (null case)',
     'theta_deg': 0.0,  'tx_m':  0.0, 'ty_m':  0.0, 'scale': 1.000},
    {'label': 'Pure rotation +5°',
     'theta_deg': 5.0,  'tx_m':  0.0, 'ty_m':  0.0, 'scale': 1.000},
    {'label': 'Pure rotation -7°',
     'theta_deg':-7.0,  'tx_m':  0.0, 'ty_m':  0.0, 'scale': 1.000},
    {'label': 'Pure translation (3m fwd)',
     'theta_deg': 0.0,  'tx_m':  3.0, 'ty_m':  0.0, 'scale': 1.000},
    {'label': 'Pure translation (2m left)',
     'theta_deg': 0.0,  'tx_m':  0.0, 'ty_m':  2.0, 'scale': 1.000},
    {'label': 'Combined: 4° + (2m, -1.5m)',
     'theta_deg': 4.0,  'tx_m':  2.0, 'ty_m': -1.5, 'scale': 1.000},
    {'label': 'Combined: -6° + (-1m, 2m)',
     'theta_deg':-6.0,  'tx_m': -1.0, 'ty_m':  2.0, 'scale': 1.000},
]



def apply_known_sim2(tensor: torch.Tensor, theta_deg: float,
                     tx_m: float, ty_m: float, scale: float = 1.0) -> torch.Tensor:
    """
    Apply a known Sim(2) transform to a BEV tensor using F.affine_grid.
    Used only to generate synthetic ground-truth targets.

    The transform is applied as an inverse warp (grid_sample convention):
    the sampling grid maps output coordinates back to input coordinates.

    Parameters
    ----------
    tensor    : [1, C, H, W]
    theta_deg : rotation in degrees (positive = CCW in BEV)
    tx_m, ty_m: translation in metres in ego BEV frame
    scale     : isotropic scale

    Returns
    -------
    warped : [1, C, H, W]
    """
    # Convert translation to normalised [-1, 1] grid units
    tx_n = tx_m / GRID_EXTENT
    ty_n = ty_m / GRID_EXTENT

    rad   = np.deg2rad(theta_deg)
    cos_t = float(np.cos(rad))
    sin_t = float(np.sin(rad))

    # Inverse affine matrix for grid_sample
    # Forward: p' = sR*p + t  →  Inverse: p = (1/s)R^T*(p' - t)
    theta_mat = torch.tensor([[
        [scale * cos_t,  scale * sin_t, -tx_n],
        [-scale * sin_t, scale * cos_t, -ty_n],
    ]], dtype=torch.float32, device=tensor.device)

    grid   = F.affine_grid(theta_mat, tensor.shape, align_corners=False)
    warped = F.grid_sample(tensor, grid, align_corners=False,
                           mode='bilinear', padding_mode='zeros')
    return warped



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading FastBEV...")
    model = FastBEV(
        in_channels=256, bev_channels=64, out_channels=256,
        num_classes=10, image_size=(256, 704), feature_size=(16, 44),
    )
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device).eval()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)
    sample_token = nusc.sample[0]['token']

    images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(nusc, sample_token)
    images           = images.unsqueeze(0).to(device)
    intrinsics       = intrinsics.unsqueeze(0).to(device)
    cam2egos         = cam2egos.unsqueeze(0).to(device)
    img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

    print("  Extracting camera BEV features...")
    with torch.no_grad():
        outputs = model(images, cam2egos, intrinsics, img_aug_matrices)
    cam_bev = outputs['bev_feat'].detach()   # [1, 256, 128, 128]
    H, W    = cam_bev.shape[2], cam_bev.shape[3]
    print(f"  cam_bev shape: {list(cam_bev.shape)}")

    print("\n" + "═" * 70)
    print("  ALIGNMENT MODULE VALIDATION — Synthetic Sim(2) Recovery")
    print("═" * 70)

    results  = []
    n_passed = 0

    for case in TEST_CASES:
        label     = case['label']
        theta_true = case['theta_deg']
        tx_m_true  = case['tx_m']
        ty_m_true  = case['ty_m']
        scale_true = case['scale']

        tx_n_true = tx_m_true / GRID_EXTENT
        ty_n_true = ty_m_true / GRID_EXTENT

        with torch.no_grad():
            target_fixed = apply_known_sim2(
                cam_bev, theta_true, tx_m_true, ty_m_true, scale_true
            )

        head      = DisplacementHead(
            camera_channels=CAMERA_CHANNELS,
            lidar_channels=CAMERA_CHANNELS,  # both sides are 256ch cam features
        ).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=OPT_LR)

        for _ in range(OPT_STEPS):
            optimizer.zero_grad()
            delta      = head(cam_bev, target_fixed)
            warped_cam = apply_dense_warp(cam_bev, delta)
            loss       = F.mse_loss(warped_cam, target_fixed)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            delta_final = head(cam_bev, target_fixed).detach()

        # The head learns the inverse warp (maps cam_bev → target_fixed),
        # so delta encodes T^{-1}. Negate to recover the forward transform T.
        # Also divide by MAX_OFFSET to rescale from [-0.1, 0.1] → [-1, 1]
        # so run_geometric_test works in full normalised coordinate units.
        MAX_OFFSET = 0.1
        delta_for_fitting = -delta_final

        params, r2, _ = run_geometric_test(delta_for_fitting, H, W)

        r2_ok    = r2                             >= R2_THRESHOLD
        theta_ok = abs(params['theta_deg'] - theta_true) <= THETA_THRESHOLD
        tx_ok    = abs(params['tx']        - tx_n_true)  <= TRANS_THRESHOLD
        ty_ok    = abs(params['ty']        - ty_n_true)  <= TRANS_THRESHOLD
        passed   = theta_ok and tx_ok and ty_ok

        if passed:
            n_passed += 1

        status = "PASS ✓" if passed else "FAIL ✗"

        print(f"\n  [{status}]  {label}")
        print(f"    {'':20s}  {'Ground truth':>14}  {'Recovered':>14}  {'OK?':>5}")
        print(f"    {'R²':20s}  {'—':>14}  {r2:14.4f}  {'✓' if r2_ok else '✗':>5}")
        print(f"    {'θ (degrees)':20s}  {theta_true:14.2f}  {params['theta_deg']:14.2f}  {'✓' if theta_ok else '✗':>5}")
        print(f"    {'tx (norm)':20s}  {tx_n_true:14.4f}  {params['tx']:14.4f}  {'✓' if tx_ok else '✗':>5}")
        print(f"    {'ty (norm)':20s}  {ty_n_true:14.4f}  {params['ty']:14.4f}  {'✓' if ty_ok else '✗':>5}")
        print(f"    {'tx (metres)':20s}  {tx_m_true:14.2f}  {params['tx']*GRID_EXTENT:14.2f}")
        print(f"    {'ty (metres)':20s}  {ty_m_true:14.2f}  {params['ty']*GRID_EXTENT:14.2f}")

        results.append({
            'label':         label,
            'passed':        passed,
            'ground_truth':  {
                'theta_deg': theta_true,
                'tx_m':      tx_m_true,
                'ty_m':      ty_m_true,
                'tx_norm':   tx_n_true,
                'ty_norm':   ty_n_true,
            },
            'recovered': {
                'theta_deg': params['theta_deg'],
                'tx_norm':   params['tx'],
                'ty_norm':   params['ty'],
                'tx_m':      params['tx'] * GRID_EXTENT,
                'ty_m':      params['ty'] * GRID_EXTENT,
                'r2':        r2,
                'geodesic':  params['geodesic_dist'],
            },
            'errors': {
                'theta_deg': abs(params['theta_deg'] - theta_true),
                'tx_norm':   abs(params['tx'] - tx_n_true),
                'ty_norm':   abs(params['ty'] - ty_n_true),
            },
        })

    print("\n" + "═" * 70)
    print(f"  RESULT: {n_passed}/{len(TEST_CASES)} cases passed")
    overall = "PIPELINE VALIDATED" if n_passed == len(TEST_CASES) else "PIPELINE NEEDS REVIEW"
    print(f"  VERDICT: {overall}")
    print("═" * 70)

    if n_passed < len(TEST_CASES):
        print("\n  Note: failing cases indicate the DisplacementHead is not")
        print("  converging on those transforms. Check OPT_STEPS and OPT_LR.")

    summary = {
        'n_passed':  n_passed,
        'n_total':   len(TEST_CASES),
        'validated': n_passed == len(TEST_CASES),
        'thresholds': {
            'r2':    R2_THRESHOLD,
            'theta': THETA_THRESHOLD,
            'trans': TRANS_THRESHOLD,
        },
        'cases': results,
    }
    out_path = OUTPUT_DIR / 'alignment_validation.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()