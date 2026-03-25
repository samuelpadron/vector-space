"""
pos_control.py — Positive control for the BEV alignment hypothesis test.

Uses the trained BEVAlignSegNet (AlignNet + SegHead) rather than a
per-sample MSE-optimised DisplacementHead. A single forward pass through
the trained model replaces the 1000-step per-sample optimisation.

Rationale
---------
The previous positive control failed because MSE between cross-modal
features (cam_bev vs lidar_bev) has no gradient signal — the features
come from different modalities and have no pixel-level correspondence.

The trained BEVAlignSegNet fixes this: its AlignNet learned to predict
geometrically meaningful offsets via segmentation supervision. The
positive control now tests whether that trained AlignNet can detect
rigid structure when it genuinely exists.

Procedure
---------
1. Load trained BEVAlignSegNet checkpoint.
2. Extract real cam_bev and lidar_bev for N samples.
3. Apply a known Sim(2) to lidar_bev.
4. Run a single forward pass through the trained model.
5. Fit Sim(2) to the predicted delta field.
6. Compare recovered parameters to ground truth.

If the hypothesis holds (R² >= 0.9, parameter errors within thresholds),
the pipeline is validated. The contrast with real data R² is then the
evidence for non-rigid misalignment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules import FastBEV, PretrainedPointPillars, load_lidar_points
from data import load_sample, load_checkpoint
from hypothesis_test import run_geometric_test
from seg_align import BEVAlignSegNet
from nuscenes.nuscenes import NuScenes

# ── Configuration ──────────────────────────────────────────────────────────────

NUSCENES_ROOT   = Path('./data/nuscenes')
FASTBEV_CKPT    = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CKPT         = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
ALIGN_CKPT      = Path('./checkpoints/align_seg_best.pth')   # trained BEVAlignSegNet
OUTPUT_DIR      = Path('./validation_output')

NUM_SAMPLES     = 8
CANVAS_SIZE     = (128, 128)

# Primary transform (supervisor's specification)
PRIMARY_TRANSFORM = {
    'theta_deg': 5.0,
    'tx_pixels': 3.0,
    'ty_pixels': 0.0,
    'scale':     1.02,
}

# Thresholds
R2_THRESHOLD    = 0.90
THETA_THRESHOLD = 1.0    # degrees
TRANS_THRESHOLD = 0.5    # pixels
SCALE_THRESHOLD = 0.01

# Sweep cases
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

NOISE_STD = 0.5   # pixels std for mixed condition


# ── Helpers ────────────────────────────────────────────────────────────────────

def apply_known_sim2(
    tensor: torch.Tensor,
    theta_deg: float,
    tx_pixels: float,
    ty_pixels: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a known Sim(2) to a BEV tensor [1, C, H, W]."""
    H, W  = tensor.shape[2], tensor.shape[3]
    tx_n  = tx_pixels / (W / 2)
    ty_n  = ty_pixels / (H / 2)
    rad   = np.deg2rad(theta_deg)
    cos_t = float(np.cos(rad))
    sin_t = float(np.sin(rad))

    theta_mat = torch.tensor([[
        [scale * cos_t,  scale * sin_t, -tx_n],
        [-scale * sin_t, scale * cos_t, -ty_n],
    ]], dtype=torch.float32, device=tensor.device)

    grid   = F.affine_grid(theta_mat, tensor.shape, align_corners=False)
    return F.grid_sample(tensor, grid, align_corners=False,
                         mode='bilinear', padding_mode='zeros')


def run_forward(
    model: BEVAlignSegNet,
    cam_bev: torch.Tensor,
    lidar_bev: torch.Tensor,
) -> tuple:
    """
    Single forward pass through the trained BEVAlignSegNet.
    Returns (params, r2, residuals) from Sim(2) fit on the predicted delta.
    """
    H, W = cam_bev.shape[2], cam_bev.shape[3]
    with torch.no_grad():
        _, delta = model(cam_bev, lidar_bev)
    params, r2, residuals = run_geometric_test(delta, H, W)
    return params, r2, residuals


def norm_to_pixels(n: float, dim: int) -> float:
    return n * (dim / 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load frozen encoders ───────────────────────────────────────────────
    print("\nLoading FastBEV (frozen)...")
    fastbev = FastBEV(
        in_channels=256, bev_channels=64, out_channels=256,
        num_classes=10, image_size=(256, 704), feature_size=(16, 44),
    )
    fastbev = load_checkpoint(fastbev, FASTBEV_CKPT, device)
    fastbev = fastbev.to(device).eval()

    print("Loading PointPillars (frozen)...")
    pointpillars = PretrainedPointPillars(
        checkpoint_path=str(PP_CKPT),
        out_spatial_size=CANVAS_SIZE,
    ).to(device).eval()

    # ── Load trained BEVAlignSegNet ────────────────────────────────────────
    print("Loading trained BEVAlignSegNet...")
    model = BEVAlignSegNet().to(device)
    ckpt  = torch.load(ALIGN_CKPT, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Loaded from epoch {ckpt['epoch']}  "
          f"train_loss={ckpt['train_loss']:.4f}  "
          f"val_loss={ckpt.get('val_loss', 'N/A')}")

    # ── Load nuScenes and extract BEV pairs ───────────────────────────────
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)

    print(f"\nExtracting BEV pairs from {NUM_SAMPLES} samples...")
    cam_bevs, lidar_bevs = [], []

    for i in range(NUM_SAMPLES):
        sample_token = nusc.sample[i]['token']
        print(f"  Sample {i}  ({sample_token[:8]}...)", end='  ')

        images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(
            nusc, sample_token
        )
        images           = images.unsqueeze(0).to(device)
        intrinsics       = intrinsics.unsqueeze(0).to(device)
        cam2egos         = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        with torch.no_grad():
            cam_bev   = fastbev(images, cam2egos, intrinsics,
                                img_aug_matrices)['bev_feat'].detach()
            raw_pts   = load_lidar_points(nusc, sample_token)
            lidar_bev = pointpillars(raw_pts, device).detach()

        cam_bevs.append(cam_bev)
        lidar_bevs.append(lidar_bev)
        print(f"cam_bev {tuple(cam_bev.shape)}  lidar_bev {tuple(lidar_bev.shape)}")

    H = cam_bevs[0].shape[2]
    W = cam_bevs[0].shape[3]

    # ── Primary positive control ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PRIMARY POSITIVE CONTROL")
    print(f"  Transform: {PRIMARY_TRANSFORM['theta_deg']}° rotation, "
          f"{PRIMARY_TRANSFORM['tx_pixels']}px translation, "
          f"scale {PRIMARY_TRANSFORM['scale']}")
    print(f"  Hypothesis: R² >= {R2_THRESHOLD},  |Δθ| <= {THETA_THRESHOLD}°, "
          f"|Δt| <= {TRANS_THRESHOLD}px")
    print("=" * 70)

    primary_results = []
    n_passed = 0

    for i, (cam_bev, lidar_bev) in enumerate(zip(cam_bevs, lidar_bevs)):
        with torch.no_grad():
            lidar_transformed = apply_known_sim2(
                lidar_bev,
                PRIMARY_TRANSFORM['theta_deg'],
                PRIMARY_TRANSFORM['tx_pixels'],
                PRIMARY_TRANSFORM['ty_pixels'],
                PRIMARY_TRANSFORM['scale'],
            )

        params, r2, _ = run_forward(model, cam_bev, lidar_transformed)

        theta_true = PRIMARY_TRANSFORM['theta_deg']
        tx_true_px = PRIMARY_TRANSFORM['tx_pixels']
        ty_true_px = PRIMARY_TRANSFORM['ty_pixels']
        scale_true = PRIMARY_TRANSFORM['scale']

        tx_rec_px  = norm_to_pixels(params['tx'], W)
        ty_rec_px  = norm_to_pixels(params['ty'], H)
        theta_err  = abs(params['theta_deg'] - theta_true)
        tx_err_px  = abs(tx_rec_px - tx_true_px)
        ty_err_px  = abs(ty_rec_px - ty_true_px)
        scale_err  = abs(params['scale'] - scale_true)

        r2_ok    = r2        >= R2_THRESHOLD
        theta_ok = theta_err <= THETA_THRESHOLD
        tx_ok    = tx_err_px <= TRANS_THRESHOLD
        ty_ok    = ty_err_px <= TRANS_THRESHOLD
        scale_ok = scale_err <= SCALE_THRESHOLD
        passed   = r2_ok and theta_ok and tx_ok and ty_ok and scale_ok

        if passed:
            n_passed += 1

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  Sample {i}  R²={r2:.4f}  "
              f"theta={params['theta_deg']:.2f}° (err {theta_err:.2f}°)  "
              f"tx={tx_rec_px:.2f}px (err {tx_err_px:.2f}px)  "
              f"scale={params['scale']:.4f} (err {scale_err:.4f})")

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

    hypothesis_holds = n_passed >= int(NUM_SAMPLES * 0.8)
    print(f"\n  PRIMARY RESULT: {n_passed}/{NUM_SAMPLES} passed")
    print(f"  VERDICT: {'HYPOTHESIS HOLDS' if hypothesis_holds else 'COUNTERFACTUAL — fix pipeline before interpreting real data'}")

    if not hypothesis_holds:
        out = {'primary': primary_results, 'hypothesis_holds': False}
        with open(OUTPUT_DIR / 'pos_control.json', 'w') as f:
            json.dump(out, f, indent=2)
        return

    # ── Sweep over transform magnitudes ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  SWEEP — operating range characterisation (sample 0 only)")
    print("=" * 70)

    sweep_results = []
    cam_bev_s0   = cam_bevs[0]
    lidar_bev_s0 = lidar_bevs[0]

    for case in SWEEP_CASES:
        with torch.no_grad():
            lidar_t = apply_known_sim2(
                lidar_bev_s0,
                case['theta_deg'], case['tx_pixels'],
                case['ty_pixels'], case['scale'],
            )
        params, r2, _ = run_forward(model, cam_bev_s0, lidar_t)
        tx_rec_px = norm_to_pixels(params['tx'], W)
        theta_err = abs(params['theta_deg'] - case['theta_deg'])
        tx_err    = abs(tx_rec_px - case['tx_pixels'])

        print(f"  {case['label']:<25s}  R²={r2:.4f}  "
              f"theta_err={theta_err:.2f}°  tx_err={tx_err:.2f}px")

        sweep_results.append({
            'label':     case['label'],
            'transform': case,
            'r2':        r2,
            'recovered': {
                'theta_deg': params['theta_deg'],
                'tx_pixels': tx_rec_px,
                'scale':     params['scale'],
            },
            'errors': {'theta_deg': theta_err, 'tx_pixels': tx_err},
        })

    # ── Mixed condition (rigid + noise) ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MIXED CONDITION — rigid transform + noise (std={NOISE_STD}px)")
    print("=" * 70)

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
            noise       = torch.randn_like(lidar_rigid) * (NOISE_STD / (W / 2))
            lidar_mixed = lidar_rigid + noise

        params, r2, _ = run_forward(model, cam_bev, lidar_mixed)
        tx_rec_px = norm_to_pixels(params['tx'], W)
        print(f"  Sample {i}:  R²={r2:.4f}  theta={params['theta_deg']:.2f}°  tx={tx_rec_px:.2f}px")
        mixed_results.append({'sample_idx': i, 'r2': r2, 'recovered': params})

    # ── Contrast: real camera-LiDAR pair (no transform) ───────────────────
    print("\n" + "=" * 70)
    print("  CONTRAST — real camera-LiDAR pair (no synthetic transform)")
    print("=" * 70)

    real_results = []
    for i, (cam_bev, lidar_bev) in enumerate(zip(cam_bevs, lidar_bevs)):
        params, r2, _ = run_forward(model, cam_bev, lidar_bev)
        tx_rec_px = norm_to_pixels(params['tx'], W)
        print(f"  Sample {i}:  R²={r2:.4f}  theta={params['theta_deg']:.2f}°  tx={tx_rec_px:.2f}px")
        real_results.append({'sample_idx': i, 'r2': r2, 'recovered': params})

    # ── Summary ────────────────────────────────────────────────────────────
    mean_r2_primary = np.mean([r['r2'] for r in primary_results])
    mean_r2_mixed   = np.mean([r['r2'] for r in mixed_results])
    mean_r2_real    = np.mean([r['r2'] for r in real_results])
    r2_drop         = mean_r2_primary - mean_r2_real

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Synthetic rigid (positive control) : mean R² = {mean_r2_primary:.4f}")
    print(f"  Rigid + noise (mixed)              : mean R² = {mean_r2_mixed:.4f}")
    print(f"  Real camera-LiDAR pair             : mean R² = {mean_r2_real:.4f}")
    print(f"  R² drop (rigid to real)            : {r2_drop:.4f}")
    print()
    if mean_r2_primary >= R2_THRESHOLD:
        print(f"  Pipeline detects rigid transforms (R²={mean_r2_primary:.3f})")
        if r2_drop > 0.5:
            print("  Large R² drop on real data supports non-rigid misalignment.")
        else:
            print("  Small R² drop — real misalignment may have partial rigid component.")
    else:
        print(f"  Pipeline does not reliably detect rigid transforms (R²={mean_r2_primary:.3f})")
        print("  Real data results remain uninterpretable.")
    print("=" * 70)

    all_results = {
        'hypothesis_holds':        hypothesis_holds,
        'primary':                 primary_results,
        'sweep':                   sweep_results,
        'mixed':                   mixed_results,
        'real_contrast':           real_results,
        'summary': {
            'mean_r2_synthetic_rigid': mean_r2_primary,
            'mean_r2_mixed':           mean_r2_mixed,
            'mean_r2_real':            mean_r2_real,
            'r2_drop_rigid_to_real':   r2_drop,
        },
    }

    out_path = OUTPUT_DIR / 'pos_control.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()