"""
run.py — entry point for the BEV alignment hypothesis test.
  1. FastBEV camera -> BEV feature extraction
  2. PointPillars LiDAR -> BEV feature extraction
  3. Per-sample optimisation of DisplacementHead (H0 proxy)
  4. Geometric Sim(2) fit + R² / geodesic distance (H1 test)
  5. Visualization
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules import (
    FastBEV,
    HandcraftedLidarBEV,
    PretrainedPointPillars,
    load_lidar_points,
    DisplacementHead,
    LidarProjector,
    apply_dense_warp,
)
from data import load_sample, load_checkpoint, decode_predictions, load_gt_boxes
from hypothesis_test import (
    run_geometric_test,
    optical_flow_sim2,
    visualize_sample,
)

# Configuration

NUSCENES_ROOT   = Path('./data/nuscenes')
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CHECKPOINT   = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
OUTPUT_DIR      = Path('./viz_output')
NUM_SAMPLES     = 10
OPT_STEPS       = 500
OPT_LR          = 1e-2

# Switch between LiDAR BEV representations:
#   'handcrafted'  — deterministic 4-channel descriptor (occupancy, height, density, intensity)
#   'pointpillars' — pretrained PointPillars FPN encoder, 256-channel learned features
LIDAR_ENCODER = 'pointpillars'   # or 'handcrafted'

# BEV grid — must be identical for camera and LiDAR branches
GRID_CONF = {'xbound': [-51.2, 51.2, 0.8], 'ybound': [-51.2, 51.2, 0.8]}


def main():
    run_output_dir = OUTPUT_DIR / LIDAR_ENCODER
    run_output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"LiDAR encoder: {LIDAR_ENCODER}")
    print(f"Output dir: {run_output_dir}")

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

    # LiDAR BEV encoder 
    if LIDAR_ENCODER == 'pointpillars':
        print("\nCreating PretrainedPointPillars encoder...")
        lidar_encoder  = PretrainedPointPillars(
            checkpoint_path  = str(PP_CHECKPOINT),
            out_spatial_size = (128, 128),
        ).to(device).eval()
        LIDAR_CHANNELS = 256   # FPN neck output channels
        print(f"  Output channels: {LIDAR_CHANNELS}")
    else:
        print("\nCreating HandcraftedLidarBEV encoder...")
        lidar_encoder  = HandcraftedLidarBEV(grid_conf=GRID_CONF)
        LIDAR_CHANNELS = HandcraftedLidarBEV.NUM_CHANNELS   # 4

    # nuScenes dataset
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)

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
        raw_points = load_lidar_points(nusc, sample_token)
        if LIDAR_ENCODER == 'pointpillars':
            print("  Building PointPillars LiDAR BEV...")
            with torch.no_grad():
                lidar_bev = lidar_encoder(raw_points, device)   # [1, 256, 128, 128]
        else:
            print("  Building hand-crafted LiDAR BEV...")
            lidar_bev = lidar_encoder.encode(raw_points, device)  # [1, 4, 128, 128]

        if lidar_bev.shape[-2:] != cam_bev.shape[-2:]:
            print(f"  Resizing LiDAR BEV {lidar_bev.shape[-2:]} → {cam_bev.shape[-2:]}")
            lidar_bev = F.interpolate(
                lidar_bev, size=cam_bev.shape[-2:], mode='bilinear', align_corners=False
            )

        print(f"  cam_bev: {tuple(cam_bev.shape)}   lidar_bev: {tuple(lidar_bev.shape)}")

        # Fresh modules per sample
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
            cam_proj   = lidar_projector(warped_cam)             # [1, 64, H, W]
            loss = F.mse_loss(cam_proj, lidar_bev.detach())
            loss.backward()
            optimizer.step()

        # Geometric hypothesis test
        H_bev, W_bev = cam_bev.shape[2], cam_bev.shape[3]
        final_delta = delta.detach()

        params, r2_score, residuals = run_geometric_test(final_delta, H_bev, W_bev)

        print(f"\n  ── Geometric Test Results ──────────────────────────")
        print(f"  R² Score         : {r2_score:.4f}  "
              f"({'H1 supported (rigid)' if r2_score > 0.85 else 'H0 supported (non-rigid)'})")
        print(f"  Geodesic Distance: {params['geodesic_dist']:.4f}")
        print(f"  Sim(2) Rotation  : {params['theta_deg']:.2f}°")
        print(f"  Sim(2) Translation: [{params['tx']:.3f}, {params['ty']:.3f}]")
        print(f"  Sim(2) Scale     : {params['scale']:.4f}")

        # Optical flow 
        print(f"\n  ── Optical Flow (Farneback) Second Opinion ─────────")
        flow_params, flow_r2, flow_residuals = optical_flow_sim2(cam_bev, lidar_bev)
        print(f"  R² Score         : {flow_r2:.4f}  "
              f"({'H1 supported (rigid)' if flow_r2 > 0.85 else 'H0 supported (non-rigid)'})")
        print(f"  Geodesic Distance: {flow_params['geodesic_dist']:.4f}")
        print(f"  Sim(2) Rotation  : {flow_params['theta_deg']:.2f}°")
        print(f"  Sim(2) Translation: [{flow_params['tx']:.3f}, {flow_params['ty']:.3f}]")
        print(f"  Sim(2) Scale     : {flow_params['scale']:.4f}")

        print(f"\n  ── Comparison ──────────────────────────────────────")
        theta_agree = abs(params['theta_deg'] - flow_params['theta_deg']) < 2.0
        r2_agree    = abs(r2_score - flow_r2) < 0.2
        print(f"  Rotation agreement  : {'✓' if theta_agree else '✗'}  "
              f"(Δθ = {abs(params['theta_deg'] - flow_params['theta_deg']):.2f}°)")
        print(f"  R² agreement        : {'✓' if r2_agree else '✗'}  "
              f"(ΔR² = {abs(r2_score - flow_r2):.3f})")

        # Decode detections for overlay
        detections = decode_predictions(outputs['predictions'], score_threshold=0.2)
        print(f"  Detections       : {len(detections)} objects above threshold")

        # Ground truth boxes for LiDAR BEV validation panel
        gt_boxes = load_gt_boxes(nusc, sample_token)
        print(f"  GT boxes         : {len(gt_boxes)} annotated objects")

        # Combined visualisation
        run_output_dir.mkdir(parents=True, exist_ok=True)
        visualize_sample(
            sample_idx  = sample_idx,
            images      = images[0],
            cam_bev     = cam_bev,
            lidar_bev   = lidar_bev,
            preds       = outputs['predictions'],
            delta       = final_delta,
            params      = params,
            residuals   = residuals,
            detections  = detections,
            gt_boxes    = gt_boxes,
            save_path   = str(run_output_dir / f"sample_{sample_idx}_combined.png"),
        )

    print(f"\n{'─'*60}")
    print(f"Done. Outputs saved to {run_output_dir}")


if __name__ == '__main__':
    main()