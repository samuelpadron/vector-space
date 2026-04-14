"""
temporal_misalignment.py — Measures how camera-LiDAR BEV spatial
correspondence (NCC) varies as a function of ego-motion (yaw rate, speed).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import FastBEV, PretrainedPointPillars, load_lidar_points
from data import load_sample, load_checkpoint
from nuscenes.nuscenes import NuScenes
from map_labels import get_ego_pose_and_location

NUSCENES_ROOT = Path('./data/nuscenes')
FASTBEV_CKPT  = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CKPT       = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
OUTPUT_DIR    = Path('./temporal_analysis')
CANVAS_SIZE   = (128, 128)
DT            = 0.5

def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = a_flat - a_flat.mean()
    b_norm = b_flat - b_flat.mean()
    denom  = np.sqrt((a_norm ** 2).sum() * (b_norm ** 2).sum())
    if denom < 1e-8:
        return 0.0
    return float((a_norm * b_norm).sum() / denom)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

print("\nLoading nuScenes...")
nusc = NuScenes(version='v1.0-mini', dataroot=str(NUSCENES_ROOT), verbose=False)

results = []

for sample_idx in range(1, len(nusc.sample)):
    token     = nusc.sample[sample_idx]['token']
    token_tm1 = nusc.sample[sample_idx - 1]['token']

    scene     = nusc.get('scene', nusc.sample[sample_idx]['scene_token'])
    scene_tm1 = nusc.get('scene', nusc.sample[sample_idx - 1]['scene_token'])
    if scene['token'] != scene_tm1['token']:
        continue

    x,     y,     theta,     _ = get_ego_pose_and_location(nusc, token)
    x_tm1, y_tm1, theta_tm1, _ = get_ego_pose_and_location(nusc, token_tm1)

    dx       = x - x_tm1
    dy       = y - y_tm1
    dtheta   = theta - theta_tm1
    speed    = np.sqrt(dx**2 + dy**2) / DT
    yaw_rate = abs(dtheta) / DT

    print(f"  Sample {sample_idx}  speed={speed:.2f}m/s  yaw_rate={yaw_rate:.2f}°/s", end='  ')

    images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(nusc, token)
    images           = images.unsqueeze(0).to(device)
    intrinsics       = intrinsics.unsqueeze(0).to(device)
    cam2egos         = cam2egos.unsqueeze(0).to(device)
    img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

    with torch.no_grad():
        cam_bev   = fastbev(images, cam2egos, intrinsics,
                            img_aug_matrices)['bev_feat'].detach()
        raw_pts   = load_lidar_points(nusc, token)
        lidar_bev = pointpillars(raw_pts, device).detach()

    cam_gray   = cam_bev[0].mean(dim=0).cpu().numpy()
    lidar_gray = lidar_bev[0].mean(dim=0).cpu().numpy()
    ncc = compute_ncc(cam_gray, lidar_gray)

    print(f"NCC={ncc:.4f}")

    results.append({
        'sample_idx': sample_idx,
        'token':      token,
        'speed':      speed,
        'yaw_rate':   yaw_rate,
        'dx':         dx,
        'dy':         dy,
        'dtheta':     dtheta,
        'ncc':        ncc,
    })

with open(OUTPUT_DIR / 'ncc_misalignment.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR / 'ncc_misalignment.json'}")

speeds    = [r['speed']    for r in results]
yaw_rates = [r['yaw_rate'] for r in results]
nccs      = [r['ncc']      for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Camera-LiDAR BEV Spatial Correspondence (NCC) vs Ego-Motion', fontsize=13)

for ax, x, xlabel, color in [
    (axes[0], speeds,    'Speed (m/s)',    'steelblue'),
    (axes[1], yaw_rates, 'Yaw Rate (°/s)', 'darkorange'),
]:
    ax.scatter(x, nccs, alpha=0.6, s=30, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('NCC (camera BEV vs LiDAR BEV)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    z = np.polyfit(x, nccs, 1)
    p = np.poly1d(z)
    x_sorted = sorted(x)
    ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.7, linewidth=1.5,
            label=f'slope={z[0]:.4f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ncc_misalignment.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to {OUTPUT_DIR / 'ncc_misalignment.png'}")
plt.close()