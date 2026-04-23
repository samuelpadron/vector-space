"""
evaluate.py — FastBEV4D NDS evaluation on nuScenes mini val.

Runs FastBEV4D with temporal fusion on the mini_val split,
saves a nuScenes-format submission, and prints NDS / mAP.

Usage:
    python scripts/evaluate.py [--checkpoint PATH]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV4D, load_checkpoint
from data import load_sample

# ── Configuration ──────────────────────────────────────────────────────────────

NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
NUSCENES_SPLIT  = 'val'             # eval on held-out val set
CHECKPOINT_PATH = Path('./checkpoints/fastbev4d/epoch_19.pth')  # trained model
EVAL_OUTPUT_DIR = Path('./eval_output')
SCORE_THRESHOLD = 0.05
MAX_DETS        = 500

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

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
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return [q.w, q.x, q.y, q.z]


def ego_to_global(det: dict, ego_pose: dict) -> dict:
    """Transform detection from ego frame to global frame."""
    t_ego = np.array(ego_pose['translation'])
    R_ego = Quaternion(ego_pose['rotation'])

    xyz_global = R_ego.rotate(np.array(det['translation'])) + t_ego
    vel_global = R_ego.rotate(np.array([det['velocity'][0], det['velocity'][1], 0.0]))
    combined_q = R_ego * Quaternion(det['rotation'])

    det['translation'] = xyz_global.tolist()
    det['rotation']    = [combined_q.w, combined_q.x, combined_q.y, combined_q.z]
    det['velocity']    = [float(vel_global[0]), float(vel_global[1])]
    return det


def decode_to_nuscenes(preds, sample_token, score_threshold=SCORE_THRESHOLD,
                        max_dets=MAX_DETS):
    """Decode CenterHead output to nuScenes submission format."""
    import torch.nn.functional as F

    task    = preds[0]
    heatmap = task['heatmap'][0].sigmoid()
    reg     = task['reg'][0]
    height  = task['height'][0]
    dim     = task['dim'][0]
    rot     = task['rot'][0]
    vel     = task['vel'][0]

    _, H, W  = heatmap.shape
    voxel_sz = 0.8

    hm_max = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    keep   = (heatmap == hm_max) & (heatmap >= score_threshold)

    detections = []
    for cls_idx in range(len(CLASS_NAMES)):
        y_idx, x_idx = torch.where(keep[cls_idx])
        for y_t, x_t in zip(y_idx, x_idx):
            y, x   = y_t.item(), x_t.item()
            score  = heatmap[cls_idx, y, x].item()
            bev_x  = (x + reg[0, y, x].item()) * voxel_sz - 51.2
            bev_y  = (y + reg[1, y, x].item()) * voxel_sz - 51.2
            bev_z  = height[0, y, x].item()
            w      = float(np.exp(np.clip(dim[0, y, x].item(), -3, 3)))
            l      = float(np.exp(np.clip(dim[1, y, x].item(), -3, 3)))
            h      = float(np.exp(np.clip(dim[2, y, x].item(), -3, 3)))
            yaw    = float(np.arctan2(rot[0, y, x].item(), rot[1, y, x].item()))
            cls_nm = CLASS_NAMES[cls_idx]
            detections.append({
                'sample_token':    sample_token,
                'translation':     [bev_x, bev_y, bev_z],
                'size':            [w, l, h],
                'rotation':        yaw_to_quaternion(yaw),
                'velocity':        [vel[0, y, x].item(), vel[1, y, x].item()],
                'detection_name':  cls_nm,
                'detection_score': score,
                'attribute_name':  DEFAULT_ATTRIBUTES[cls_nm],
            })

    detections.sort(key=lambda d: d['detection_score'], reverse=True)
    return detections[:max_dets]


def build_scene_sequence(nusc, val_tokens):
    """
    Group val tokens by scene and sort each scene chronologically.
    Returns list of scenes, each a list of sample tokens in time order.
    """
    scene_map = defaultdict(list)
    token_set = set(val_tokens)

    for sample in nusc.sample:
        if sample['token'] in token_set:
            scene_map[sample['scene_token']].append(sample)

    scenes = []
    for scene_samples in scene_map.values():
        # Sort by timestamp
        scene_samples.sort(key=lambda s: s['timestamp'])
        scenes.append([s['token'] for s in scene_samples])

    return scenes


def compute_se2(ego_prev: dict, ego_curr: dict, grid_res: float = 0.8) -> torch.Tensor:
    """SE(2) between two nuScenes ego_pose dicts. Returns [1, 3] tensor."""
    t_prev = np.array(ego_prev['translation'])
    t_curr = np.array(ego_curr['translation'])
    q_prev = Quaternion(ego_prev['rotation'])
    q_curr = Quaternion(ego_curr['rotation'])

    dx_m  = t_curr[0] - t_prev[0]
    dy_m  = t_curr[1] - t_prev[1]
    dyaw  = q_curr.yaw_pitch_roll[0] - q_prev.yaw_pitch_roll[0]
    dyaw  = (dyaw + np.pi) % (2 * np.pi) - np.pi

    return torch.tensor([[dx_m / grid_res, dy_m / grid_res, dyaw]],
                        dtype=torch.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading FastBEV4D...")
    model = FastBEV4D(
        in_channels=256,
        bev_channels=64,
        out_channels=256,
        num_classes=10,
        image_size=(256, 704),
        feature_size=(16, 44),
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    # Support both raw state_dict and training checkpoint format
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    model = model.to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── nuScenes ──────────────────────────────────────────────────────────────
    print(f"\nLoading nuScenes {NUSCENES_VER}...")
    nusc       = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    val_scenes = set(create_splits_scenes()[NUSCENES_SPLIT])
    val_tokens = [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in val_scenes
    ]
    print(f"  Val samples: {len(val_tokens)}")

    scenes = build_scene_sequence(nusc, val_tokens)

    # ── Inference with temporal context ───────────────────────────────────────
    print("\nRunning inference...")
    all_results = {}
    total = sum(len(s) for s in scenes)
    done  = 0

    for scene_tokens in scenes:
        bev_feat_prev = None
        ego_pose_prev = None

        for token in scene_tokens:
            done += 1
            print(f"  [{done:3d}/{total}]", end='\r')

            # load_sample already returns [1, 3, H, W] / [1, 4, 4] / [1, 3, 3]
            img, intr, c2e, _, ego_pose, _ = load_sample(nusc, token)
            img  = img.to(device)
            intr = intr.to(device)
            c2e  = c2e.to(device)

            # SE(2) from previous frame (None on scene-first frame)
            se2 = None
            if ego_pose_prev is not None:
                se2 = compute_se2(ego_pose_prev, ego_pose).to(device)

            with torch.no_grad():
                outputs = model(
                    img, c2e, intr,
                    bev_feat_prev=bev_feat_prev,
                    se2=se2,
                )

            # Cache pre-fusion encoder output as prev for next frame (BEVDet4D convention)
            bev_feat_prev = outputs['bev_feat_enc'].detach()
            ego_pose_prev = ego_pose

            # Decode and transform to global frame
            dets = decode_to_nuscenes(outputs['predictions'], token)
            dets = [ego_to_global(d, ego_pose) for d in dets]
            all_results[token] = dets

    print(f"\n  Total detections: {sum(len(v) for v in all_results.values())}")

    # ── Save submission ────────────────────────────────────────────────────────
    submission = {
        'meta': {
            'use_camera': True, 'use_lidar': False,
            'use_radar':  False, 'use_map':   False, 'use_external': False,
        },
        'results': all_results,
    }
    submission_path = EVAL_OUTPUT_DIR / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f)
    print(f"Saved submission → {submission_path}")

    # ── NuScenes eval ─────────────────────────────────────────────────────────
    print("\nRunning nuScenes evaluation...")
    cfg      = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(
        nusc,
        config=cfg,
        result_path=str(submission_path),
        eval_set=NUSCENES_SPLIT,
        output_dir=str(EVAL_OUTPUT_DIR),
        verbose=False,
    )
    metrics, _ = evaluator.evaluate()

    print("\n" + "=" * 55)
    print("  FastBEV4D — nuScenes mini val")
    print("=" * 55)
    print(f"  NDS  : {metrics.nd_score:.4f}")
    print(f"  mAP  : {metrics.mean_ap:.4f}")
    print()
    print("  Per-class AP:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<25s} {metrics.mean_dist_aps.get(cls, 0.0):.4f}")
    print("=" * 55)

    summary = {
        'nds': metrics.nd_score,
        'map': metrics.mean_ap,
        'per_class_ap': {c: metrics.mean_dist_aps.get(c, 0.0) for c in CLASS_NAMES},
    }
    with open(EVAL_OUTPUT_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved → {EVAL_OUTPUT_DIR / 'metrics_summary.json'}")


if __name__ == '__main__':
    main()
