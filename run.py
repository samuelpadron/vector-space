"""
Entry point for the BEV alignment hypothesis test:
  1. FastBEV camera → BEV feature extraction
  2. PointPillars LiDAR → BEV feature extraction
  3. Per-sample optimisation of DisplacementHead (H0 proxy)
  4. Geometric Sim(2) fit + R² / geodesic distance (H1 test)
  5. Visualisation: quiver comparison + residual heatmap
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent / 'src')) 

from src.modules import (
    FastBEV,
    HandcraftedLidarBEV,
    load_lidar_points,
    DisplacementHead,
    LidarProjector,
    apply_dense_warp,
)
from src.data import load_sample, load_checkpoint
from src.hypothesis_test import (
    run_geometric_test,
    visualize_geometry_comparison,
    visualize_residual_heatmap,
)

# ── Configuration ─────────────────────────────────────────────────────────────

NUSCENES_ROOT   = Path('./data/nuscenes')
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
OUTPUT_DIR      = Path('./viz_output')
NUM_SAMPLES     = 10
OPT_STEPS       = 500
OPT_LR          = 1e-2
LIDAR_CHANNELS  = 4

# BEV grid — must be identical for camera and LiDAR branches
GRID_CONF = {'xbound': [-51.2, 51.2, 0.8], 'ybound': [-51.2, 51.2, 0.8]}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── FastBEV camera model ──────────────────────────────────────────────
    print("\nCreating FastBEV model...")
    model = FastBEV(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )
    if CHECKPOINT_PATH.exists():
        model = load_checkpoint(model, CHECKPOINT_PATH, device)
    else:
        print(f"  Warning: checkpoint not found at {CHECKPOINT_PATH}")
        print("  Running with random weights (for structural testing only).")

    model = model.to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── PointPillars LiDAR encoder ────────────────────────────────────────
    print("\nCreating LidarBEV...")
    lidar_encoder = HandcraftedLidarBEV(
        grid_conf=GRID_CONF,
    )
    # Weights are random-but-fixed; used as a deterministic feature extractor.
    # The hypothesis test measures displacement field structure, not absolute
    # feature quality.

    # ── nuScenes ──────────────────────────────────────────────────────────
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)

    # ── Per-sample loop ───────────────────────────────────────────────────
    for sample_idx in range(min(NUM_SAMPLES, len(nusc.sample))):
        sample = nusc.sample[sample_idx]
        sample_token = sample['token']
        print(f"\n{'─'*60}")
        print(f"Sample {sample_idx}  ({sample_token[:8]}...)")

        # Camera BEV
        images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(nusc, sample_token)
        images           = images.unsqueeze(0).to(device)
        intrinsics       = intrinsics.unsqueeze(0).to(device)
        cam2egos         = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        print(f"  Running FastBEV inference  (input: {images.shape})...")
        with torch.no_grad():
            outputs = model(images, cam2egos, intrinsics, img_aug_matrices)
            cam_bev = outputs['bev_feat']   # [1, 256, H, W]

        # LiDAR BEV
        print("  Encoding LiDAR with PointPillars...")
        raw_points = load_lidar_points(nusc, sample_token)
        with torch.no_grad():
            lidar_bev = lidar_encoder.encode(raw_points, device)   # [1, 4, 128, 128]

        if lidar_bev.shape[-2:] != cam_bev.shape[-2:]:
            print(f"  Resizing LiDAR BEV {lidar_bev.shape[-2:]} → {cam_bev.shape[-2:]}")
            lidar_bev = F.interpolate(
                lidar_bev, size=cam_bev.shape[-2:], mode='bilinear', align_corners=False
            )

        print(f"  cam_bev: {tuple(cam_bev.shape)}   lidar_bev: {tuple(lidar_bev.shape)}")

        # Fresh modules per sample — each experiment is independent
        displacement_head = DisplacementHead(
            camera_channels=cam_bev.shape[1],
            lidar_channels=LIDAR_CHANNELS,
        ).to(device)
        lidar_projector = LidarProjector(
            camera_channels=cam_bev.shape[1],
            lidar_channels=LIDAR_CHANNELS,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(displacement_head.parameters()) + list(lidar_projector.parameters()),
            lr=OPT_LR,
        )

        # Optimisation loop
        print(f"  Optimising DisplacementHead ({OPT_STEPS} steps)...")
        for step in range(OPT_STEPS):
            optimizer.zero_grad()
            delta      = displacement_head(cam_bev, lidar_bev)   # [1, 2, H, W]
            warped_cam = apply_dense_warp(cam_bev, delta)        # [1, 256, H, W]
            lidar_occupancy = lidar_bev[:, 0:1, :, :]          # channel 0 = occupancy
            valid_mask = (lidar_occupancy > 0).float()          # 1 where LiDAR has returns
            cam_proj = lidar_projector(warped_cam)
            loss = F.mse_loss(cam_proj * valid_mask, lidar_bev.detach() * valid_mask)
            loss.backward()
            optimizer.step()

        # Geometric hypothesis test
        H_bev, W_bev = cam_bev.shape[2], cam_bev.shape[3]
        final_delta = delta.detach()

        params, r2_score, residuals = run_geometric_test(final_delta, H_bev, W_bev)

        print(f"\n  ── Geometric Test Results ──────────────────────────")
        print(f"  R² Score         : {r2_score:.4f}  ")
        print(f"  Geodesic Distance: {params['geodesic_dist']:.4f}")
        print(f"  Sim(2) Rotation  : {params['theta_deg']:.2f}°")
        print(f"  Sim(2) Translation: [{params['tx']:.3f}, {params['ty']:.3f}]")
        print(f"  Sim(2) Scale     : {params['scale']:.4f}")

        # Visualise
        disp_dir = OUTPUT_DIR / "displacement"
        heat_dir = OUTPUT_DIR / "heatmaps"
        disp_dir.mkdir(parents=True, exist_ok=True)
        heat_dir.mkdir(parents=True, exist_ok=True)

        visualize_geometry_comparison(
            final_delta, params, sample_idx,
            save_path=str(disp_dir / f"sample_{sample_idx}_geometry_comparison.png"),
        )
        visualize_residual_heatmap(
            residuals, sample_idx,
            save_path=str(heat_dir / f"sample_{sample_idx}_residual_heatmap.png"),
        )

    print(f"\n{'─'*60}")
    print(f"Done. Outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
