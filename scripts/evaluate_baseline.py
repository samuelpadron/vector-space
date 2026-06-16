"""
evaluate_baseline.py — FastBEV4D in single-frame mode (no temporal fusion).

Uses the same fine-tuned checkpoint as the temporal model but always passes
bev_feat_prev=None, forcing the fusion conv to act as a pass-through.
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

# Configuration

NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
NUSCENES_SPLIT  = 'val'
CHECKPOINT_PATH = Path('./checkpoints/fastbev4d_fusion/best.pth')
EVAL_OUTPUT_DIR = Path('./eval_output_baseline')
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

# Helper functions
def yaw_to_quaternion(yaw: float) -> list:
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return [q.w, q.x, q.y, q.z]


def ego_to_global(det: dict, ego_pose: dict) -> dict:
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
    scene_map = defaultdict(list)
    token_set = set(val_tokens)

    for sample in nusc.sample:
        if sample['token'] in token_set:
            scene_map[sample['scene_token']].append(sample)

    scenes = []
    for scene_samples in scene_map.values():
        scene_samples.sort(key=lambda s: s['timestamp'])
        scenes.append([s['token'] for s in scene_samples])

    return scenes


def main():
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading FastBEV4D (single-frame mode, bev_feat_prev=None)...")
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

    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)

    model = model.to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nLoading nuScenes {NUSCENES_VER}...")
    nusc       = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    val_scenes = set(create_splits_scenes()[NUSCENES_SPLIT])
    val_tokens = [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in val_scenes
    ]
    print(f"  Val samples: {len(val_tokens)}")

    scenes = build_scene_sequence(nusc, val_tokens)

    print("\nRunning inference...")
    all_results = {}
    total = sum(len(s) for s in scenes)
    done  = 0

    for scene_tokens in scenes:
        for token in scene_tokens:
            done += 1
            print(f"  [{done:3d}/{total}]", end='\r')

            img, intr, c2e, _, ego_pose, _ = load_sample(nusc, token)
            img  = img.unsqueeze(0).to(device)   # [1, 1, 3, H, W]
            intr = intr.unsqueeze(0).to(device)  # [1, 1, 3, 3]
            c2e  = c2e.unsqueeze(0).to(device)   # [1, 1, 4, 4]

            with torch.no_grad():
                outputs = model(img, c2e, intr, bev_feats_prev=None, se2_list=None)

            dets = decode_to_nuscenes(outputs['predictions'], token)
            dets = [ego_to_global(d, ego_pose) for d in dets]
            all_results[token] = dets

    print(f"\n  Total detections: {sum(len(v) for v in all_results.values())}")

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
    print("  FastBEV4D (no fusion) — nuScenes val")
    print("=" * 55)
    print(f"  NDS  : {metrics.nd_score:.4f}")
    print(f"  mAP  : {metrics.mean_ap:.4f}")
    print()
    print("  Per-class AP:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<25s} {metrics.mean_dist_aps.get(cls, 0.0):.4f}")
    print("=" * 55)

    tp = metrics.tp_errors  # keys: trans_err, scale_err, orient_err, vel_err, attr_err
    summary = {
        'nds': metrics.nd_score,
        'map': metrics.mean_ap,
        'tp_errors': {
            'mATE': tp.get('trans_err',  float('nan')),
            'mASE': tp.get('scale_err',  float('nan')),
            'mAOE': tp.get('orient_err', float('nan')),
            'mAVE': tp.get('vel_err',    float('nan')),
            'mAAE': tp.get('attr_err',   float('nan')),
        },
        'per_class_ap': {c: metrics.mean_dist_aps.get(c, 0.0) for c in CLASS_NAMES},
    }
    with open(EVAL_OUTPUT_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  mATE : {summary['tp_errors']['mATE']:.4f}  (translation, m)")
    print(f"  mASE : {summary['tp_errors']['mASE']:.4f}  (scale, 1-IoU)")
    print(f"  mAOE : {summary['tp_errors']['mAOE']:.4f}  (orientation, rad)")
    print(f"  mAVE : {summary['tp_errors']['mAVE']:.4f}  (velocity, m/s)")
    print(f"  mAAE : {summary['tp_errors']['mAAE']:.4f}  (attribute)")
    print("=" * 55)
    print(f"Metrics saved → {EVAL_OUTPUT_DIR / 'metrics_summary.json'}")


if __name__ == '__main__':
    main()
