"""
evaluate_fastbev_r50_cbgs.py — FastBEV-R50-CBGS evaluation on nuScenes val.

Loads the pretrained fastbev-r50-cbgs checkpoint and runs single-frame
inference with the full 6-camera rig (no temporal fusion).

Usage:
    python scripts/evaluate_fastbev_r50_cbgs.py [--checkpoint PATH] [--reeval]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from pyquaternion import Quaternion
from PIL import Image
from torchvision.transforms.functional import normalize
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV, load_checkpoint

# ── Configuration ──────────────────────────────────────────────────────────────

NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
NUSCENES_SPLIT  = 'val'
SCORE_THRESHOLD = 0.05
MAX_DETS        = 500
IMAGE_SIZE      = (256, 704)   # (H, W) — matches fastbev-r50-cbgs training config

# nuScenes 6-camera rig in the canonical FastBEV order
CAMERAS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]

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

# ── Argument parsing ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='FastBEV-R50-CBGS 6-cam evaluation')
    parser.add_argument(
        '--checkpoint', type=Path,
        default=Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth'),
        help='Path to pretrained fastbev-r50-cbgs checkpoint.',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('./eval_output/eval_output_r50_cbgs'),
        help='Evaluation output directory.',
    )
    parser.add_argument(
        '--reeval', action='store_true',
        help='Skip inference; re-evaluate existing submission.json in --output-dir.',
    )
    return parser.parse_args()

# ── Data loading ────────────────────────────────────────────────────────────────

def load_sample_multicam(nusc, sample_token, target_size=IMAGE_SIZE):
    """
    Load all 6 cameras for a nuScenes sample.

    Returns
    -------
    images    : FloatTensor [6, 3, H, W]  ImageNet-normalised
    intrinsics: FloatTensor [6, 3, 3]
    cam2egos  : FloatTensor [6, 4, 4]
    ego_pose  : dict with 'translation' and 'rotation' keys
    """
    sample = nusc.get('sample', sample_token)

    imgs, intrs, c2es = [], [], []
    for cam in CAMERAS:
        cam_token = sample['data'][cam]
        cam_data  = nusc.get('sample_data', cam_token)

        img = Image.open(Path(nusc.dataroot) / cam_data['filename']).convert('RGB')
        orig_w, orig_h = img.size

        img_t = torch.from_numpy(
            np.array(img.resize((target_size[1], target_size[0])))
        ).permute(2, 0, 1).float() / 255.0
        img_t = normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        imgs.append(img_t)

        cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        K  = np.array(cs['camera_intrinsic'])
        K[0, :] *= target_size[1] / orig_w
        K[1, :] *= target_size[0] / orig_h

        c2e      = np.eye(4)
        c2e[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        c2e[:3, 3]  = np.array(cs['translation'])

        intrs.append(torch.from_numpy(K).float())
        c2es.append(torch.from_numpy(c2e).float())

    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    return (
        torch.stack(imgs),   # [6, 3, H, W]
        torch.stack(intrs),  # [6, 3, 3]
        torch.stack(c2es),   # [6, 4, 4]
        ego_pose,
    )

# ── Detection decoding ──────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return [q.w, q.x, q.y, q.z]


def ego_to_global(det, ego_pose):
    t_ego = np.array(ego_pose['translation'])
    R_ego = Quaternion(ego_pose['rotation'])

    xyz_global = R_ego.rotate(np.array(det['translation'])) + t_ego
    vel_global = R_ego.rotate(np.array([det['velocity'][0], det['velocity'][1], 0.0]))
    combined_q = R_ego * Quaternion(det['rotation'])

    det['translation'] = xyz_global.tolist()
    det['rotation']    = [combined_q.w, combined_q.x, combined_q.y, combined_q.z]
    det['velocity']    = [float(vel_global[0]), float(vel_global[1])]
    return det


def decode_to_nuscenes(preds, sample_token, score_threshold=SCORE_THRESHOLD, max_dets=MAX_DETS):
    import torch.nn.functional as F

    task    = preds[0]
    heatmap = task['heatmap'][0].sigmoid()
    reg     = task['reg'][0]
    height  = task['height'][0]
    dim     = task['dim'][0]
    rot     = task['rot'][0]
    vel     = task['vel'][0]

    voxel_sz = 0.8
    hm_max   = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    keep     = (heatmap == hm_max) & (heatmap >= score_threshold)

    detections = []
    for cls_idx in range(len(CLASS_NAMES)):
        y_idx, x_idx = torch.where(keep[cls_idx])
        for y_t, x_t in zip(y_idx, x_idx):
            y, x   = y_t.item(), x_t.item()
            score  = heatmap[cls_idx, y, x].item()
            bev_x  = (x + reg[0, y, x].item()) * voxel_sz - 51.6
            bev_y  = (y + reg[1, y, x].item()) * voxel_sz - 51.6
            bev_z  = height[0, y, x].item()
            l      = float(np.exp(np.clip(dim[0, y, x].item(), -3, 3)))
            w      = float(np.exp(np.clip(dim[1, y, x].item(), -3, 3)))
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

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir  : {args.output_dir}")
    print(f"\nLoading nuScenes {NUSCENES_VER}...")
    nusc       = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    val_scenes = set(create_splits_scenes()[NUSCENES_SPLIT])
    val_tokens = [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in val_scenes
    ]
    print(f"  Val samples : {len(val_tokens)}")

    submission_path = args.output_dir / 'submission.json'

    if args.reeval:
        if not submission_path.exists():
            raise FileNotFoundError(f"No submission.json in {args.output_dir}")
        print(f"Skipping inference — reusing {submission_path}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device      : {device}")
        print(f"Cameras     : {len(CAMERAS)}  ({', '.join(CAMERAS)})")
        print(f"Checkpoint  : {args.checkpoint}")

        print("\nBuilding FastBEV (single-frame, 6-cam)...")
        model = FastBEV(
            in_channels=256,
            bev_channels=64,
            out_channels=256,
            num_classes=10,
            image_size=IMAGE_SIZE,
            feature_size=(IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16),
        )

        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state, strict=False)

        model = model.to(device).eval()
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Collect val tokens in scene-chronological order (scenes × frames)
        scene_map = defaultdict(list)
        token_set = set(val_tokens)
        for s in nusc.sample:
            if s['token'] in token_set:
                scene_map[s['scene_token']].append(s)
        ordered_tokens = []
        for scene_samples in scene_map.values():
            scene_samples.sort(key=lambda s: s['timestamp'])
            ordered_tokens.extend(t['token'] for t in scene_samples)

        print("\nRunning inference...")
        all_results = {}
        total = len(ordered_tokens)

        for i, token in enumerate(ordered_tokens):
            print(f"  [{i + 1:4d}/{total}]", end='\r')

            imgs, intrs, c2es, ego_pose = load_sample_multicam(nusc, token)
            imgs   = imgs.unsqueeze(0).to(device)    # [1, 6, 3, H, W]
            intrs  = intrs.unsqueeze(0).to(device)   # [1, 6, 3, 3]
            c2es   = c2es.unsqueeze(0).to(device)    # [1, 6, 4, 4]

            with torch.no_grad():
                outputs = model(imgs, c2es, intrs)

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
        with open(submission_path, 'w') as f:
            json.dump(submission, f)
        print(f"Saved submission → {submission_path}")

    print("\nRunning nuScenes evaluation...")
    cfg       = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(
        nusc,
        config=cfg,
        result_path=str(submission_path),
        eval_set=NUSCENES_SPLIT,
        output_dir=str(args.output_dir),
        verbose=False,
    )
    metrics, _ = evaluator.evaluate()

    print("\n" + "=" * 55)
    print("  FastBEV-R50-CBGS (6-cam) — nuScenes val")
    print("=" * 55)
    print(f"  NDS  : {metrics.nd_score:.4f}")
    print(f"  mAP  : {metrics.mean_ap:.4f}")
    print()
    print("  Per-class AP:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<25s} {metrics.mean_dist_aps.get(cls, 0.0):.4f}")
    print("=" * 55)

    tp = metrics.tp_errors
    summary = {
        'model':    'fastbev-r50-cbgs',
        'cameras':  CAMERAS,
        'nds':      metrics.nd_score,
        'map':      metrics.mean_ap,
        'tp_errors': {
            'mATE': tp.get('trans_err',  float('nan')),
            'mASE': tp.get('scale_err',  float('nan')),
            'mAOE': tp.get('orient_err', float('nan')),
            'mAVE': tp.get('vel_err',    float('nan')),
            'mAAE': tp.get('attr_err',   float('nan')),
        },
        'per_class_ap': {c: metrics.mean_dist_aps.get(c, 0.0) for c in CLASS_NAMES},
    }
    with open(args.output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  mATE : {summary['tp_errors']['mATE']:.4f}  (translation, m)")
    print(f"  mASE : {summary['tp_errors']['mASE']:.4f}  (scale, 1-IoU)")
    print(f"  mAOE : {summary['tp_errors']['mAOE']:.4f}  (orientation, rad)")
    print(f"  mAVE : {summary['tp_errors']['mAVE']:.4f}  (velocity, m/s)")
    print(f"  mAAE : {summary['tp_errors']['mAAE']:.4f}  (attribute)")
    print("=" * 55)
    print(f"Metrics saved → {args.output_dir / 'metrics_summary.json'}")


if __name__ == '__main__':
    main()
