"""
evaluate.py — FastBEV detection baseline validation on nuScenes mini.

Runs the FastBEV detection head on the nuScenes mini val split, saves
predictions in the official nuScenes submission format, and computes
NDS and mAP using the nuScenes devkit evaluator.

Purpose
-------
This script validates that the pretrained FastBEV model produces
geometrically meaningful camera BEV features by checking that detection
metrics are in the expected ballpark of the published results. Without
this check, the downstream hypothesis test results (R², residual
heatmaps) cannot be trusted, since they assume the camera BEV features
encode real scene geometry.

Expected results
----------------
FastBEV (camera-only) on nuScenes full val reports NDS ≈ 0.35–0.38.
On nuScenes mini (81 val samples) expect lower absolute numbers due
to the small sample size, but the model should clearly outperform
random predictions and detect the major object categories.

Usage
-----
    python evaluate.py

Output
------
    eval_output/
        submission.json          nuScenes-format predictions
        metrics_summary.json     NDS, mAP, per-class AP
        metrics_details.json     full per-class breakdown
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV
from data import load_sample, load_checkpoint

# ── Configuration ──────────────────────────────────────────────────────────────

NUSCENES_ROOT   = Path('./data/nuscenes')
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
EVAL_OUTPUT_DIR = Path('./eval_output')
NUSCENES_VER    = 'v1.0-mini'
SCORE_THRESHOLD = 0.05    # Lower than usual to capture more detections on mini
MAX_DETS        = 500    # nuScenes allows up to 500 detections per sample

# nuScenes class names in CenterHead order (must match checkpoint training order)
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Default attributes per class — required by nuScenes evaluator.
# For a camera-only model without attribute prediction, we use the most
# common attribute per category as a fixed default.
DEFAULT_ATTRIBUTES = {
    'car':                  'vehicle.parked',
    'truck':                'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'bus':                  'vehicle.parked',
    'trailer':              'vehicle.parked',
    'barrier':              '',
    'motorcycle':           'cycle.without_rider',
    'bicycle':              'cycle.without_rider',
    'pedestrian':           'pedestrian.standing',
    'traffic_cone':         '',
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> list:
    """
    Convert a yaw angle (rotation around z-axis) to a nuScenes quaternion
    [w, x, y, z].  nuScenes uses the convention where the vehicle points
    along the x-axis at yaw=0.
    """
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return [q.w, q.x, q.y, q.z]


def get_ego_pose(nusc: NuScenes, sample_token: str) -> dict:
    """
    Return the ego pose record for a given sample.
    The ego pose gives the vehicle's position and orientation in the
    global (map) frame at the time of the sample's LIDAR_TOP sweep.
    """
    sample    = nusc.get('sample', sample_token)
    lidar_sd  = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    return ego_pose


def ego_to_global(det: dict, ego_pose: dict) -> dict:
    """
    Transform a single detection from ego frame to global frame in-place.

    nuScenes ego pose gives:
      translation : [x, y, z]  — ego position in global frame
      rotation    : [w, x, y, z] quaternion — ego orientation in global frame

    For each detection:
      global_xyz  = R_ego * det_xyz + t_ego
      global_yaw  = det_yaw + ego_yaw   (add ego heading)
      global_vel  = R_ego * det_vel     (rotate velocity vector)
    """
    t_ego = np.array(ego_pose['translation'])           # (3,)
    R_ego = Quaternion(ego_pose['rotation'])             # pyquaternion

    # Rotate + translate position
    xyz_ego    = np.array(det['translation'])            # (3,)
    xyz_global = R_ego.rotate(xyz_ego) + t_ego

    # Rotate velocity (2-D, embed in 3-D with z=0)
    vel_ego    = np.array([det['velocity'][0], det['velocity'][1], 0.0])
    vel_global = R_ego.rotate(vel_ego)

    # Add ego yaw to detection yaw
    # Extract yaw from detection quaternion, add ego yaw, re-encode
    det_q      = Quaternion(det['rotation'])
    combined_q = R_ego * det_q

    det['translation'] = xyz_global.tolist()
    det['rotation']    = [combined_q.w, combined_q.x, combined_q.y, combined_q.z]
    det['velocity']    = [float(vel_global[0]), float(vel_global[1])]
    return det


def decode_to_nuscenes(preds, sample_token: str,
                        score_threshold: float = SCORE_THRESHOLD,
                        max_dets: int = MAX_DETS) -> list:
    """
    Decode CenterHead predictions into nuScenes submission format.

    Key differences from the visualisation decode_predictions():
    - Returns velocity [vx, vy] from the vel head
    - Converts yaw to quaternion
    - Uses nuScenes class name strings
    - Applies proper BEV coordinate convention

    Parameters
    ----------
    preds        : raw CenterHead output (list of task dicts)
    sample_token : nuScenes sample token for this frame
    score_threshold, max_dets : filtering parameters

    Returns
    -------
    list of dicts in nuScenes detection submission format
    """
    import torch.nn.functional as F

    task = preds[0]

    heatmap = task['heatmap'][0].sigmoid()   # (C, H, W)
    reg     = task['reg'][0]                 # (2, H, W)
    height  = task['height'][0]              # (1, H, W)
    dim     = task['dim'][0]                 # (3, H, W)
    rot     = task['rot'][0]                 # (2, H, W) — sin, cos
    vel     = task['vel'][0]                 # (2, H, W)

    num_classes, H, W = heatmap.shape
    voxel_size = 0.8   # metres per BEV pixel

    # Local-max NMS
    heatmap_max = F.max_pool2d(
        heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1
    )[0]
    keep = (heatmap == heatmap_max) & (heatmap >= score_threshold)

    detections = []

    for cls_idx in range(num_classes):
        y_idx, x_idx = torch.where(keep[cls_idx])
        for y_t, x_t in zip(y_idx, x_idx):
            y, x = y_t.item(), x_t.item()
            score = heatmap[cls_idx, y, x].item()

            # BEV position in ego frame (metres)
            # nuScenes BEV: x = forward, y = left
            bev_x = (x + reg[0, y, x].item()) * voxel_size - 51.2
            bev_y = (y + reg[1, y, x].item()) * voxel_size - 51.2
            bev_z = height[0, y, x].item()

            # Dimensions — CenterHead predicts log(dim)
            w = float(np.exp(np.clip(dim[0, y, x].item(), -3, 3)))
            l = float(np.exp(np.clip(dim[1, y, x].item(), -3, 3)))
            h = float(np.exp(np.clip(dim[2, y, x].item(), -3, 3)))

            # Yaw — atan2(sin, cos)
            yaw = float(np.arctan2(rot[0, y, x].item(), rot[1, y, x].item()))

            # Velocity in ego frame
            vx = vel[0, y, x].item()
            vy = vel[1, y, x].item()

            cls_name = CLASS_NAMES[cls_idx]

            detections.append({
                'sample_token':     sample_token,
                'translation':      [bev_x, bev_y, bev_z],
                'size':             [w, l, h],
                'rotation':         yaw_to_quaternion(yaw),
                'velocity':         [vx, vy],
                'detection_name':   cls_name,
                'detection_score':  score,
                'attribute_name':   DEFAULT_ATTRIBUTES[cls_name],
            })

    # Sort by score, keep top-N
    detections.sort(key=lambda d: d['detection_score'], reverse=True)
    return detections[:max_dets]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nCreating FastBEV model...")
    model = FastBEV(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Cannot validate baseline without pretrained weights."
        )

    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load nuScenes ─────────────────────────────────────────────────────
    print(f"\nLoading nuScenes {NUSCENES_VER}...")
    nusc = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)

    # Get val split tokens
    # nuScenes mini val split is defined in the splits file
    from nuscenes.utils.splits import create_splits_scenes
    splits     = create_splits_scenes()
    val_scenes = set(splits['mini_val'])

    val_tokens = [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in val_scenes
    ]
    print(f"  Val samples: {len(val_tokens)}")

    # ── Run inference ─────────────────────────────────────────────────────
    print("\nRunning inference on val set...")
    all_results = {}

    for i, sample_token in enumerate(val_tokens):
        print(f"  [{i+1:3d}/{len(val_tokens)}] {sample_token[:8]}...", end='\r')

        images, intrinsics, cam2egos, img_aug_matrices, ego_pose, _ = load_sample(
            nusc, sample_token
        )
        images           = images.unsqueeze(0).to(device)
        intrinsics       = intrinsics.unsqueeze(0).to(device)
        cam2egos         = cam2egos.unsqueeze(0).to(device)
        img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(images, cam2egos, intrinsics, img_aug_matrices)

        # Decode in ego frame then transform to global frame
        dets = decode_to_nuscenes(
            outputs['predictions'], sample_token,
            score_threshold=SCORE_THRESHOLD,
        )
        dets = [ego_to_global(d, ego_pose) for d in dets]
        all_results[sample_token] = dets

    print(f"\n  Done. Total detections: {sum(len(v) for v in all_results.values())}")

    # ── Save submission JSON ───────────────────────────────────────────────
    submission = {
        "meta": {
            "use_camera":   True,
            "use_lidar":    False,
            "use_radar":    False,
            "use_map":      False,
            "use_external": False,
        },
        "results": all_results,
    }

    submission_path = EVAL_OUTPUT_DIR / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f)
    print(f"\nSaved submission → {submission_path}")

    # ── Run nuScenes evaluation ────────────────────────────────────────────
    print("\nRunning nuScenes evaluation...")
    cfg = config_factory('detection_cvpr_2019')

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=str(submission_path),
        eval_set='mini_val',
        output_dir=str(EVAL_OUTPUT_DIR),
        verbose=True,
    )

    metrics, metric_data_list = nusc_eval.evaluate()

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  FASTBEV BASELINE VALIDATION — nuScenes mini val")
    print("═" * 55)
    print(f"  NDS  : {metrics.nd_score:.4f}")
    print(f"  mAP  : {metrics.mean_ap:.4f}")
    print()
    print("  Per-class AP:")
    for cls_name in CLASS_NAMES:
        ap = metrics.mean_dist_aps.get(cls_name, 0.0)
        print(f"    {cls_name:<25s} {ap:.4f}")
    print("═" * 55)
    print()
    print("  Published FastBEV (camera-only, full val): NDS ≈ 0.35–0.38")
    print("  Note: lower absolute numbers expected on mini val (81 samples).")
    print("  The model is valid if major categories (car, pedestrian) show")
    print("  non-trivial AP and NDS is clearly above random (≈ 0.0).")
    print("═" * 55)

    # Save summary for reference
    summary = {
        'nds':  metrics.nd_score,
        'map':  metrics.mean_ap,
        'per_class_ap': {
            cls: metrics.mean_dist_aps.get(cls, 0.0)
            for cls in CLASS_NAMES
        },
        'note': (
            'Evaluated on nuScenes mini val (81 samples). '
            'Lower than full val results expected due to dataset size.'
        ),
    }
    with open(EVAL_OUTPUT_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved → {EVAL_OUTPUT_DIR / 'metrics_summary.json'}")


    with open('eval_output/submission.json') as f:
        sub = json.load(f)

    all_dets = [d for dets in sub['results'].values() for d in dets]
    print(Counter(d['detection_name'] for d in all_dets))
    print(f"Total: {len(all_dets)}")
    

if __name__ == '__main__':
    main()