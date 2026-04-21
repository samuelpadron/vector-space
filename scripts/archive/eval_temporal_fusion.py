"""
eval_temporal_fusion.py — Idea A: YOLO-gated selective temporal fusion.

Evaluates three fusion strategies on training samples using the existing
frozen FastBEV + trained AlignNet+SegHead checkpoint. No retraining.

Conditions
----------
  none   : single frame (t only) — baseline
  naive  : average features from t-1 and t equally for all pixels
  yolo   : YOLO-gated — average static regions, keep t-only for dynamic

The three conditions share identical weights (frozen FastBEV + best AlignNet
checkpoint). The only difference is what goes into the BEV transformer.
Since the checkpoint was trained on single-frame features, naive and yolo
conditions are slightly out-of-distribution — this gives a zero-shot signal.
If gated fusion helps here, retraining with gated fusion will be even stronger.

Usage
-----
    python scripts/eval_temporal_fusion.py [--samples 200] [--yolo-model yolo11x.pt]

Output
------
    temporal_analysis/fusion_eval/
        results.json       per-sample mIoU for each condition
        summary.json       mean / std mIoU across conditions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV, PretrainedPointPillars, load_lidar_points
from modules.temporal_fusion import build_dynamic_mask, gated_fuse, naive_fuse
from modules.seg_align import BEVAlignSegNet
from data import load_sample, load_checkpoint, load_raw_image, get_triplet_tokens
from map_labels import load_nusc_maps, get_bev_map_labels, NUM_CLASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NUSCENES_ROOT = Path('./data/nuscenes')
FASTBEV_CKPT  = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CKPT       = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
SEG_CKPT      = Path('./checkpoints/align_seg_best.pth')
OUT_DIR       = Path('./temporal_analysis/fusion_eval')

BEV_EXTENT  = 51.2
CANVAS_SIZE = (128, 128)
CAM_NAMES   = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_iou(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = (logits.sigmoid() > 0.5).float()
    inter = (preds * labels).sum(dim=(0, 2, 3))
    union = (preds + labels).clamp(max=1).sum(dim=(0, 2, 3))
    return (inter / (union + 1e-6)).cpu()


def fastbev_forward_with_feats(fastbev, img_feats, cam2ego, intrinsics, img_aug):
    """Continue FastBEV forward pass from pre-extracted image features."""
    bev_feat, depth = fastbev.img_view_transformer(
        img_feats, cam2ego, intrinsics, img_aug
    )
    bev_feats = fastbev.img_bev_encoder_backbone(bev_feat)
    bev_feat  = fastbev.img_bev_encoder_neck(bev_feats)
    return bev_feat


def get_train_triplets(nusc: NuScenes):
    split_scenes = set(create_splits_scenes()['mini_train'])
    triplets = []
    for sample in nusc.sample:
        if nusc.get('scene', sample['scene_token'])['name'] not in split_scenes:
            continue
        t = get_triplet_tokens(nusc, sample['token'])
        if t is not None:
            triplets.append(t)
    return triplets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',    type=int,   default=None,
                        help='Max triplets to evaluate (default: all)')
    parser.add_argument('--yolo-model', default='yolo11x.pt')
    parser.add_argument('--yolo-conf',  type=float, default=0.25)
    parser.add_argument('--version',    default='v1.0-mini')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("\nLoading FastBEV (frozen)...")
    fastbev = FastBEV(in_channels=256, bev_channels=64, out_channels=256,
                      num_classes=10, image_size=(256, 704), feature_size=(16, 44))
    fastbev = load_checkpoint(fastbev, FASTBEV_CKPT, device)
    fastbev = fastbev.to(device).eval()

    print("Loading PointPillars (frozen)...")
    pointpillars = PretrainedPointPillars(
        checkpoint_path=str(PP_CKPT), out_spatial_size=CANVAS_SIZE
    ).to(device).eval()

    print("Loading AlignNet+SegHead checkpoint...")
    seg_ckpt = torch.load(SEG_CKPT, map_location=device, weights_only=False)
    seg_model = BEVAlignSegNet().to(device)
    seg_model.load_state_dict(seg_ckpt['model'])
    seg_model.eval()

    print(f"Loading YOLO: {args.yolo_model} ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"\nLoading nuScenes {args.version}...")
    nusc      = NuScenes(version=args.version, dataroot=str(NUSCENES_ROOT), verbose=False)
    nusc_maps = load_nusc_maps(str(NUSCENES_ROOT))

    triplets = get_train_triplets(nusc)
    if args.samples:
        triplets = triplets[:args.samples]
    print(f"Evaluating {len(triplets)} triplets...")

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    CLASS_NAMES = ['lane_divider', 'ped_crossing', 'road_segment']
    results = []

    iou_acc = {cond: torch.zeros(NUM_CLASSES) for cond in ('none', 'naive', 'yolo')}

    with torch.no_grad():
        for idx, (tok_tm1, tok_t, _) in enumerate(triplets):

            # ----- Load FastBEV inputs for t and t-1 -----
            imgs_t, intrs, c2e, aug, _ = load_sample(nusc, tok_t)
            imgs_tm1, _, _, _, _       = load_sample(nusc, tok_tm1)

            imgs_t   = imgs_t.unsqueeze(0).to(device)
            imgs_tm1 = imgs_tm1.unsqueeze(0).to(device)
            intrs    = intrs.unsqueeze(0).to(device)
            c2e      = c2e.unsqueeze(0).to(device)
            aug      = aug.unsqueeze(0).to(device)

            # ----- Extract backbone features -----
            feat_t   = fastbev.extract_img_feat(imgs_t)    # [1, 6, 256, 16, 44]
            feat_tm1 = fastbev.extract_img_feat(imgs_tm1)  # [1, 6, 256, 16, 44]

            # ----- LiDAR BEV (same for all conditions) -----
            lidar_bev = pointpillars(load_lidar_points(nusc, tok_t), device)

            # ----- Map labels at t -----
            labels = get_bev_map_labels(
                nusc, nusc_maps, tok_t,
                bev_extent=BEV_EXTENT, canvas_size=CANVAS_SIZE, device=device,
            ).unsqueeze(0)

            # ----- YOLO dynamic mask for gated fusion -----
            raw_imgs = [load_raw_image(nusc, tok_t, cam) for cam in CAM_NAMES]
            dyn_mask = build_dynamic_mask(yolo, raw_imgs, conf=args.yolo_conf,
                                          device=device)   # [6, 1, 16, 44]

            sample_ious = {}

            for condition in ('none', 'naive', 'yolo'):
                if condition == 'none':
                    feats = feat_t
                elif condition == 'naive':
                    feats = naive_fuse(feat_tm1, feat_t)
                else:  # yolo
                    feats = gated_fuse(feat_tm1, feat_t, dyn_mask)

                cam_bev = fastbev_forward_with_feats(fastbev, feats, c2e, intrs, aug)
                logits, _ = seg_model(cam_bev, lidar_bev)
                iou = compute_iou(logits, labels)

                iou_acc[condition] += iou
                sample_ious[condition] = {
                    'miou':   float(iou.mean()),
                    'per_class': iou.tolist(),
                }

            results.append({'idx': idx, 'tok_t': tok_t, **sample_ious})

            if (idx + 1) % 20 == 0:
                n = idx + 1
                print(f"  [{n:3d}/{len(triplets)}]  "
                      f"none={iou_acc['none'].mean()/n:.4f}  "
                      f"naive={iou_acc['naive'].mean()/n:.4f}  "
                      f"yolo={iou_acc['yolo'].mean()/n:.4f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n = len(triplets)
    summary = {}
    print('\n' + '=' * 60)
    print(f"  Temporal Fusion Evaluation  ({n} training triplets)")
    print('=' * 60)
    print(f"  {'Condition':<10}  {'mIoU':>8}  " +
          "  ".join(f"{c:>14}" for c in CLASS_NAMES))
    print('  ' + '-' * 56)

    for cond in ('none', 'naive', 'yolo'):
        mean_iou = iou_acc[cond] / n
        miou     = mean_iou.mean().item()
        summary[cond] = {
            'miou':      miou,
            'per_class': {CLASS_NAMES[i]: float(mean_iou[i]) for i in range(NUM_CLASSES)},
        }
        print(f"  {cond:<10}  {miou:>8.4f}  " +
              "  ".join(f"{mean_iou[i].item():>14.4f}" for i in range(NUM_CLASSES)))

    print('=' * 60)

    # Delta vs. no-fusion baseline
    base = summary['none']['miou']
    for cond in ('naive', 'yolo'):
        delta = summary[cond]['miou'] - base
        print(f"  {cond} vs none: {delta:+.4f}")

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
