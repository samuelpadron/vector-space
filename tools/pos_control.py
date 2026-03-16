"""
positive_control.py — Positive control for the BEV alignment hypothesis test.

"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import (
    FastBEV, HandcraftedLidarBEV, PretrainedPointPillars,
    load_lidar_points, DisplacementHead, LidarProjector, apply_dense_warp,
)
from data import load_sample, load_checkpoint
from hypothesis_test import run_geometric_test
from nuscenes.nuscenes import NuScenes


NUSCENES_ROOT   = Path('./data/nuscenes')
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CHECKPOINT   = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
OUTPUT_DIR      = Path('./validation_output')
GRID_CONF       = {'xbound': [-51.2, 51.2, 0.8], 'ybound': [-51.2, 51.2, 0.8]}
GRID_EXTENT     = 51.2   # metres - half-width of BEV
PIXEL_SIZE      = 0.8    # metres per BEV pixel

OPT_STEPS       = 1000   # more steps — real cross-modal pairs are harder
OPT_LR          = 1e-2
NUM_SAMPLES     = 8     

# Use PointPillars as LiDAR encoder (256ch matches cam_bev — no projector needed)
# Set to 'handcrafted' to use the 4-channel encoder instead
LIDAR_ENCODER   = 'pointpillars'

PRIMARY_TRANSFORM = {
    'theta_deg': 5.0,
    'tx_pixels': 3.0,
    'ty_pixels': 0.0,
    'scale':     1.02,
}

R2_THRESHOLD    = 0.90
THETA_THRESHOLD = 1.0    # degrees
TRANS_THRESHOLD = 0.5    # pixels
SCALE_THRESHOLD = 0.01

# Sweep configuration
RUN_SWEEP = True
SWEEP_CASES = [
    {'theta_deg':  1.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '1° rotation'},
    {'theta_deg':  5.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '5° rotation'},
    {'theta_deg': 10.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '10° rotation'},
    {'theta_deg': 20.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '20° rotation'},
    {'theta_deg':  0.0, 'tx_pixels': 1.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '1px translation'},
    {'theta_deg':  0.0, 'tx_pixels': 3.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '3px translation'},
    {'theta_deg':  0.0, 'tx_pixels': 5.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '5px translation'},
    {'theta_deg':  0.0, 'tx_pixels':10.0, 'ty_pixels': 0.0, 'scale': 1.00, 'label': '10px translation'},
    {'theta_deg':  0.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 0.95, 'label': 'scale 0.95'},
    {'theta_deg':  0.0, 'tx_pixels': 0.0, 'ty_pixels': 0.0, 'scale': 1.05, 'label': 'scale 1.05'},
    {'theta_deg':  5.0, 'tx_pixels': 3.0, 'ty_pixels': 0.0, 'scale': 1.02, 'label': 'primary (combined)'},
]

# Mixed condition noise level (pixels std)
NOISE_STD = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def apply_known_sim2(
    tensor: torch.Tensor,
    theta_deg: float,
    tx_pixels: float,
    ty_pixels: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply a known Sim(2) to a BEV tensor [1, C, H, W].
    Translation specified in pixels for interpretability.
    Uses F.affine_grid + F.grid_sample (inverse warp convention).
    """
    H, W = tensor.shape[2], tensor.shape[3]

    # Convert pixel translation to normalised [-1, 1] units
    tx_n = tx_pixels / (W / 2)
    ty_n = ty_pixels / (H / 2)

    rad   = np.deg2rad(theta_deg)
    cos_t = float(np.cos(rad))
    sin_t = float(np.sin(rad))

    theta_mat = torch.tensor([[
        [scale * cos_t,  scale * sin_t, -tx_n],
        [-scale * sin_t, scale * cos_t, -ty_n],
    ]], dtype=torch.float32, device=tensor.device)

    grid   = F.affine_grid(theta_mat, tensor.shape, align_corners=False)
    warped = F.grid_sample(tensor, grid, align_corners=False,
                           mode='bilinear', padding_mode='zeros')
    return warped


def run_alignment(
    cam_bev: torch.Tensor,
    lidar_bev_transformed: torch.Tensor,
    device: torch.device,
    opt_steps: int = OPT_STEPS,
    label: str = '',
) -> dict:
    """
    Run DisplacementHead optimisation and Sim(2) fitting on a cross-modal pair.
    Returns recovered parameters and R².
    """
    H, W = cam_bev.shape[2], cam_bev.shape[3]
    cam_ch   = cam_bev.shape[1]
    lidar_ch = lidar_bev_transformed.shape[1]

    head      = DisplacementHead(
        camera_channels=cam_ch,
        lidar_channels=lidar_ch,
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=OPT_LR)

    for step in range(opt_steps):
        optimizer.zero_grad()
        delta      = head(cam_bev, lidar_bev_transformed)
        warped_cam = apply_dense_warp(cam_bev, delta)
        loss       = F.mse_loss(warped_cam, lidar_bev_transformed)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta_final       = head(cam_bev, lidar_bev_transformed).detach()
    delta_for_fitting = -delta_final   # head learns inverse warp

    params, r2, residuals = run_geometric_test(delta_for_fitting, H, W)
    return params, r2, residuals


def pixels_to_norm(px: float, dim: int) -> float:
    return px / (dim / 2)


def norm_to_pixels(n: float, dim: int) -> float:
    return n * (dim / 2)


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

    if LIDAR_ENCODER == 'pointpillars':
        print("Loading PretrainedPointPillars encoder...")
        lidar_enc = PretrainedPointPillars(
            checkpoint_path=str(PP_CHECKPOINT),
            out_spatial_size=(128, 128),
        ).to(device).eval()
        lidar_ch = 256
    else:
        lidar_enc = HandcraftedLidarBEV(grid_conf=GRID_CONF)
        lidar_ch  = HandcraftedLidarBEV.NUM_CHANNELS

    print(f"LiDAR encoder: {LIDAR_ENCODER}  ({lidar_ch} channels)")

    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)

    print(f"\nExtracting BEV pairs from {NUM_SAMPLES} samples...")
    cam_bevs   = []
    lidar_bevs = []

    for i in range(NUM_SAMPLES):
        sample_token = nusc.sample[i]['token']
        print(f"  Sample {i}  ({sample_token[:8]}...)", end='  ')

        images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(nusc, sample_token)
        images           = images.unsqueeze(0).to(device)
        intrinsics       = intrinsics.unsqueeze(0).to(device)
        cam2egos         = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(images, cam2egos, intrinsics, img_aug_matrices)
        cam_bev = outputs['bev_feat'].detach()

        raw_points = load_lidar_points(nusc, sample_token)
        if LIDAR_ENCODER == 'pointpillars':
            with torch.no_grad():
                lidar_bev = lidar_enc(raw_points, device).detach()
        else:
            lidar_bev = lidar_enc.encode(raw_points, device).detach()

        cam_bevs.append(cam_bev)
        lidar_bevs.append(lidar_bev)
        print(f"cam_bev {tuple(cam_bev.shape)}  lidar_bev {tuple(lidar_bev.shape)}")

    H = cam_bevs[0].shape[2]
    W = cam_bevs[0].shape[3]

    print("\n" + "═" * 70)
    print("  PRIMARY POSITIVE CONTROL TEST")
    print(f"  Transform: {PRIMARY_TRANSFORM['theta_deg']}° rotation, "
          f"{PRIMARY_TRANSFORM['tx_pixels']} pixel translation, "
          f"scale {PRIMARY_TRANSFORM['scale']}")
    print(f"  Hypothesis: R² ≥ {R2_THRESHOLD},  |Δθ| ≤ {THETA_THRESHOLD}°, "
          f"|Δt| ≤ {TRANS_THRESHOLD}px, |Δs| ≤ {SCALE_THRESHOLD}")
    print("═" * 70)

    primary_results = []
    n_passed = 0

    for i, (cam_bev, lidar_bev) in enumerate(zip(cam_bevs, lidar_bevs)):
        print(f"\n  Sample {i}  optimising ({OPT_STEPS} steps)...")

        with torch.no_grad():
            lidar_transformed = apply_known_sim2(
                lidar_bev,
                PRIMARY_TRANSFORM['theta_deg'],
                PRIMARY_TRANSFORM['tx_pixels'],
                PRIMARY_TRANSFORM['ty_pixels'],
                PRIMARY_TRANSFORM['scale'],
            )

        params, r2, _ = run_alignment(cam_bev, lidar_transformed, device)

        # True values in comparable units
        theta_true = PRIMARY_TRANSFORM['theta_deg']
        tx_true_px = PRIMARY_TRANSFORM['tx_pixels']
        ty_true_px = PRIMARY_TRANSFORM['ty_pixels']
        scale_true = PRIMARY_TRANSFORM['scale']

        # Recovered translation in pixels
        tx_rec_px  = norm_to_pixels(params['tx'], W)
        ty_rec_px  = norm_to_pixels(params['ty'], H)

        theta_err  = abs(params['theta_deg'] - theta_true)
        tx_err_px  = abs(tx_rec_px - tx_true_px)
        ty_err_px  = abs(ty_rec_px - ty_true_px)
        scale_err  = abs(params['scale'] - scale_true)

        r2_ok      = r2         >= R2_THRESHOLD
        theta_ok   = theta_err  <= THETA_THRESHOLD
        tx_ok      = tx_err_px  <= TRANS_THRESHOLD
        ty_ok      = ty_err_px  <= TRANS_THRESHOLD
        scale_ok   = scale_err  <= SCALE_THRESHOLD
        passed     = r2_ok and theta_ok and tx_ok and ty_ok and scale_ok

        if passed:
            n_passed += 1

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}]  R²={r2:.4f}  θ={params['theta_deg']:.2f}°(err {theta_err:.2f}°)  "
              f"tx={tx_rec_px:.2f}px(err {tx_err_px:.2f})  scale={params['scale']:.4f}(err {scale_err:.4f})")

        primary_results.append({
            'sample_idx': i,
            'passed':     passed,
            'r2':         r2,
            'ground_truth': {
                'theta_deg': theta_true,
                'tx_pixels': tx_true_px,
                'ty_pixels': ty_true_px,
                'scale':     scale_true,
            },
            'recovered': {
                'theta_deg': params['theta_deg'],
                'tx_pixels': tx_rec_px,
                'ty_pixels': ty_rec_px,
                'scale':     params['scale'],
            },
            'errors': {
                'theta_deg': theta_err,
                'tx_pixels': tx_err_px,
                'ty_pixels': ty_err_px,
                'scale':     scale_err,
            },
        })

    print(f"\n  PRIMARY RESULT: {n_passed}/{NUM_SAMPLES} samples passed")
    hypothesis_holds = n_passed >= int(NUM_SAMPLES * 0.8)   # 80% pass rate
    print(f"  VERDICT: {'HYPOTHESIS HOLDS — pipeline validated' if hypothesis_holds else 'COUNTERFACTUAL — pipeline needs review'}")

    if not hypothesis_holds:
        print("\n  *** STOP: do not interpret low R² on real data until pipeline is fixed ***")
        # Still save and exit — don't run extensions
        results = {'primary': primary_results, 'hypothesis_holds': hypothesis_holds}
        with open(OUTPUT_DIR / 'positive_control.json', 'w') as f:
            json.dump(results, f, indent=2)
        return

    if RUN_SWEEP:
        print("\n" + "═" * 70)
        print("  EXTENSION 1: SWEEP — operating range characterisation")
        print("  Using sample 0 only for speed")
        print("═" * 70)

        cam_bev_s0   = cam_bevs[0]
        lidar_bev_s0 = lidar_bevs[0]
        sweep_results = []

        for case in SWEEP_CASES:
            with torch.no_grad():
                lidar_t = apply_known_sim2(
                    lidar_bev_s0,
                    case['theta_deg'], case['tx_pixels'],
                    case['ty_pixels'], case['scale'],
                )

            params, r2, _ = run_alignment(cam_bev_s0, lidar_t, device,
                                           opt_steps=OPT_STEPS, label=case['label'])

            tx_rec_px = norm_to_pixels(params['tx'], W)
            ty_rec_px = norm_to_pixels(params['ty'], H)
            theta_err = abs(params['theta_deg'] - case['theta_deg'])
            tx_err    = abs(tx_rec_px - case['tx_pixels'])

            print(f"  {case['label']:<25s}  R²={r2:.4f}  "
                  f"θ_err={theta_err:.2f}°  tx_err={tx_err:.2f}px")

            sweep_results.append({
                'label':      case['label'],
                'transform':  case,
                'r2':         r2,
                'recovered':  {
                    'theta_deg': params['theta_deg'],
                    'tx_pixels': tx_rec_px,
                    'ty_pixels': ty_rec_px,
                    'scale':     params['scale'],
                },
                'errors': {
                    'theta_deg': theta_err,
                    'tx_pixels': tx_err,
                },
            })

    print("\n" + "═" * 70)
    print("  EXTENSION 2: MIXED CONDITION — rigid transform + per-pixel noise")
    print(f"  Noise std: {NOISE_STD} pixels, same primary transform")
    print("═" * 70)

    mixed_results = []
    for i, (cam_bev, lidar_bev) in enumerate(zip(cam_bevs[:3], lidar_bevs[:3])):
        with torch.no_grad():
            lidar_rigid = apply_known_sim2(
                lidar_bev,
                PRIMARY_TRANSFORM['theta_deg'],
                PRIMARY_TRANSFORM['tx_pixels'],
                PRIMARY_TRANSFORM['ty_pixels'],
                PRIMARY_TRANSFORM['scale'],
            )
            # Add small per-pixel displacement noise to simulate "mostly rigid"
            noise = torch.randn_like(lidar_rigid) * (NOISE_STD / (W / 2))
            lidar_mixed = lidar_rigid + noise

        params, r2, _ = run_alignment(cam_bev, lidar_mixed, device, label=f'mixed_s{i}')
        tx_rec_px = norm_to_pixels(params['tx'], W)

        print(f"  Sample {i}:  R²={r2:.4f}  θ={params['theta_deg']:.2f}°  tx={tx_rec_px:.2f}px")
        mixed_results.append({'sample_idx': i, 'r2': r2, 'recovered': params})

    print("\n" + "═" * 70)
    print("  EXTENSION 3: CONTRAST — real camera-LiDAR pair (no transform)")
    print("  This is the actual experimental condition from the hypothesis test.")
    print("  Direct comparison of R² here vs. primary positive control.")
    print("═" * 70)

    real_results = []
    for i, (cam_bev, lidar_bev) in enumerate(zip(cam_bevs, lidar_bevs)):
        params, r2, _ = run_alignment(cam_bev, lidar_bev, device, label=f'real_s{i}')
        tx_rec_px = norm_to_pixels(params['tx'], W)

        print(f"  Sample {i}:  R²={r2:.4f}  θ={params['theta_deg']:.2f}°  tx={tx_rec_px:.2f}px")
        real_results.append({'sample_idx': i, 'r2': r2, 'recovered': params})

    mean_r2_primary = np.mean([r['r2'] for r in primary_results])
    mean_r2_mixed   = np.mean([r['r2'] for r in mixed_results])
    mean_r2_real    = np.mean([r['r2'] for r in real_results])

    print("\n" + "═" * 70)
    print("  SUMMARY — R² COMPARISON ACROSS CONDITIONS")
    print("═" * 70)
    print(f"  Synthetic rigid (positive control) : mean R² = {mean_r2_primary:.4f}")
    print(f"  Rigid + noise (mixed)              : mean R² = {mean_r2_mixed:.4f}")
    print(f"  Real camera-LiDAR pair             : mean R² = {mean_r2_real:.4f}")
    print()
    print("  Interpretation:")
    if mean_r2_primary >= R2_THRESHOLD:
        print(f"  ✓ Pipeline recovers rigid transforms (R²={mean_r2_primary:.3f} ≥ {R2_THRESHOLD})")
    else:
        print(f"  ✗ Pipeline fails to recover rigid transforms (R²={mean_r2_primary:.3f} < {R2_THRESHOLD})")
    r2_drop = mean_r2_primary - mean_r2_real
    print(f"  R² drop from synthetic rigid to real pair: {r2_drop:.3f}")
    if r2_drop > 0.5:
        print("  → Large drop supports genuine non-rigid misalignment in real data.")
    else:
        print("  → Small drop; real misalignment may have partial rigid component.")
    print("═" * 70)

    all_results = {
        'config': {
            'lidar_encoder':      LIDAR_ENCODER,
            'num_samples':        NUM_SAMPLES,
            'opt_steps':          OPT_STEPS,
            'primary_transform':  PRIMARY_TRANSFORM,
            'thresholds': {
                'r2':    R2_THRESHOLD,
                'theta': THETA_THRESHOLD,
                'trans': TRANS_THRESHOLD,
                'scale': SCALE_THRESHOLD,
            },
        },
        'hypothesis_holds':   hypothesis_holds,
        'primary':            primary_results,
        'sweep':              sweep_results if RUN_SWEEP else [],
        'mixed':              mixed_results,
        'real_contrast':      real_results,
        'summary': {
            'mean_r2_synthetic_rigid': mean_r2_primary,
            'mean_r2_mixed':           mean_r2_mixed,
            'mean_r2_real':            mean_r2_real,
            'r2_drop_rigid_to_real':   r2_drop,
        },
    }

    out_path = OUTPUT_DIR / 'positive_control.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()