"""
train_align_with_prediction.py — Idea 2 training.

Extends train_align.py by feeding a predicted t+1 dynamic-agent BEV
occupancy channel into AlignNet alongside cam_bev and lidar_bev.

The extra channel tells AlignNet approximately where dynamic objects
will be at t+1, giving it a spatial prior to help compensate for the
camera-LiDAR timing offset in dynamic regions.

Architecture change
-------------------
AlignNet input: cat(cam_bev[256], lidar_bev[256], pred_bev[1]) = 513 ch
Everything else is identical to train_align.py.

Two velocity modes (--velocity)
---------------------------------
  gt   : Oracle. Uses nuScenes GT box positions + annotated velocities.
         No YOLO. Upper bound — tells us the best possible gain from
         knowing dynamic object positions.

  yolo : Realistic. Runs YOLO on frames t-1 and t to detect objects,
         matches them via IoU, estimates velocity from BEV displacement,
         and predicts t+1 positions via constant velocity.

Training data
-------------
Only samples that have both a previous and next frame (triplets) are
used, so scene-boundary samples are excluded. This is a small reduction
compared to train_align.py (~5% fewer samples for nuScenes).

Usage
-----
  # Oracle upper bound
  python scripts/train_align_with_prediction.py --velocity gt

  # Realistic YOLO-estimated velocity
  python scripts/train_align_with_prediction.py --velocity yolo --yolo-model yolo11x.pt

  # Ablation: disable alignment, keep prediction channel
  python scripts/train_align_with_prediction.py --velocity gt --no-align

Outputs
-------
  checkpoints/pred_{gt|yolo}_best.pth   — best checkpoint by val loss
  checkpoints/pred_{gt|yolo}_last.pth
  train_log_pred_{gt|yolo}.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV, PretrainedPointPillars, load_lidar_points
from data import load_sample, load_checkpoint, load_raw_image, get_triplet_tokens
from data.nuscenes_loader import get_sensor_transforms
from modules.yolo_bev_predictor import predict_bev_gt, predict_bev_yolo
from map_labels import load_nusc_maps, get_bev_map_labels, NUM_CLASSES
from modules.seg_align import BEVAlignSegNet, seg_loss, compute_pos_weights

# ---------------------------------------------------------------------------
# Paths and hyper-parameters
# ---------------------------------------------------------------------------
NUSCENES_ROOT  = Path('./data/nuscenes')
FASTBEV_CKPT   = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CKPT        = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
CKPT_DIR       = Path('./checkpoints')

NUSCENES_VER   = 'v1.0-mini'
BEV_EXTENT     = 51.2
CANVAS_SIZE    = (128, 128)
PRED_CHANNELS  = 1          # single occupancy channel

EPOCHS         = 30
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
LOG_EVERY      = 50
VAL_EVERY      = 5
PRED_DROPOUT   = 0.4   # probability of zeroing pred_bev during training

CAM_CHANNELS   = 256
LIDAR_CHANNELS = 256
MID_CHANNELS   = 128
MAX_OFFSET     = 0.1

# Camera used for YOLO lifting (front gives the largest BEV coverage)
LIFT_CAM       = 'CAM_FRONT'
DT             = 0.5       # nuScenes sample interval, seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_iou(logits, labels, threshold=0.5):
    preds = (logits.sigmoid() > threshold).float()
    inter = (preds * labels).sum(dim=(0, 2, 3))
    union = (preds + labels).clamp(max=1).sum(dim=(0, 2, 3))
    return (inter / (union + 1e-6)).cpu()


def get_triplets_for_split(nusc: NuScenes, split: str):
    """
    Return all valid (tok_tm1, tok_t, tok_tp1) triplets in a nuScenes split.
    Scene-boundary samples (no prev or next) are excluded automatically.
    """
    split_scenes = set(create_splits_scenes()[split])
    triplets = []
    for sample in nusc.sample:
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        if scene_name not in split_scenes:
            continue
        result = get_triplet_tokens(nusc, sample['token'])
        if result is not None:
            triplets.append(result)
    return triplets


def build_pred_bev(
    nusc,
    tok_tm1: str,
    tok_t:   str,
    velocity_mode: str,
    yolo_model,
    yolo_conf: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the predicted t+1 occupancy BEV tensor for one training sample.

    Parameters
    ----------
    tok_tm1        : sample token at t-1  (only used in 'yolo' mode)
    tok_t          : sample token at t    (source of velocity / GT boxes)
    velocity_mode  : 'gt' or 'yolo'
    yolo_model     : loaded YOLO model    (None in 'gt' mode)
    yolo_conf      : YOLO confidence threshold
    device         : target device

    Returns
    -------
    FloatTensor [1, 1, 128, 128]  — unsqueezed batch-dim, on `device`
    """
    if velocity_mode == 'gt':
        pred = predict_bev_gt(nusc, tok_t, dt=DT)

    else:  # 'yolo'
        # Raw images at native resolution (YOLO expects un-normalised input)
        img_tm1 = load_raw_image(nusc, tok_tm1, LIFT_CAM)
        img_t   = load_raw_image(nusc, tok_t,   LIFT_CAM)

        # Camera calibration at native resolution for ground-plane lift
        sample_t    = nusc.get('sample', tok_t)
        cam_tok     = sample_t['data'][LIFT_CAM]
        intrinsic_native, cam2ego = get_sensor_transforms(nusc, cam_tok)

        pred = predict_bev_yolo(
            yolo_model, img_tm1, img_t,
            intrinsic_native, cam2ego,
            conf=yolo_conf, dt=DT,
        )

    return pred.unsqueeze(0).to(device)   # [1, 1, 128, 128]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train AlignNet with predicted BEV occupancy')
    p.add_argument('--velocity',    choices=['gt', 'yolo'], default='gt',
                   help='Velocity mode: gt (oracle) or yolo (estimated)')
    p.add_argument('--yolo-model',  default='yolo11x.pt',
                   help='YOLO weights (used only when --velocity yolo)')
    p.add_argument('--yolo-conf',   type=float, default=0.25)
    p.add_argument('--no-align',    action='store_true',
                   help='Ablation: disable warp, keep pred_bev channel')
    p.add_argument('--version',     default=NUSCENES_VER,
                   help='nuScenes version string')
    return p.parse_args()


def main():
    args = parse_args()

    tag       = f'pred_{args.velocity}' + ('_no_align' if args.no_align else '')
    ckpt_best = CKPT_DIR / f'{tag}_best.pth'
    ckpt_last = CKPT_DIR / f'{tag}_last.pth'
    log_path  = Path(f'train_log_{tag}.json')

    print(f"Run: {tag}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Frozen encoders
    # -----------------------------------------------------------------------
    print("\nLoading FastBEV (frozen)...")
    fastbev = FastBEV(
        in_channels=256, bev_channels=64, out_channels=256,
        num_classes=10, image_size=(256, 704), feature_size=(16, 44),
    )
    fastbev = load_checkpoint(fastbev, FASTBEV_CKPT, device)
    fastbev = fastbev.to(device).eval()
    for p in fastbev.parameters():
        p.requires_grad_(False)

    print("Loading PointPillars (frozen)...")
    pointpillars = PretrainedPointPillars(
        checkpoint_path=str(PP_CKPT),
        out_spatial_size=CANVAS_SIZE,
    ).to(device).eval()
    for p in pointpillars.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # YOLO (only in yolo mode)
    # -----------------------------------------------------------------------
    yolo_model = None
    if args.velocity == 'yolo':
        from ultralytics import YOLO
        print(f"Loading YOLO: {args.yolo_model} ...")
        yolo_model = YOLO(args.yolo_model)

    # -----------------------------------------------------------------------
    # Trainable model  (pred_channels=1 → AlignNet sees 513-ch input)
    # -----------------------------------------------------------------------
    print(f"\nCreating BEVAlignSegNet (pred_channels={PRED_CHANNELS})...")
    model = BEVAlignSegNet(
        camera_channels=CAM_CHANNELS,
        lidar_channels=LIDAR_CHANNELS,
        mid_channels=MID_CHANNELS,
        max_offset=MAX_OFFSET,
        no_align=args.no_align,
        pred_channels=PRED_CHANNELS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10,
    )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"\nLoading nuScenes {args.version} ...")
    nusc      = NuScenes(version=args.version, dataroot=str(NUSCENES_ROOT), verbose=False)
    nusc_maps = load_nusc_maps(str(NUSCENES_ROOT))

    train_split = 'mini_train' if 'mini' in args.version else 'train'
    val_split   = 'mini_val'   if 'mini' in args.version else 'val'

    train_triplets = get_triplets_for_split(nusc, train_split)
    val_triplets   = get_triplets_for_split(nusc, val_split)
    print(f"  Training triplets : {len(train_triplets)}")
    print(f"  Validation triplets: {len(val_triplets)}")

    # Class weights from t+1 labels of training triplets
    print("\nComputing class weights...")
    all_labels = [
        get_bev_map_labels(nusc, nusc_maps, tok_tp1,
                           bev_extent=BEV_EXTENT,
                           canvas_size=CANVAS_SIZE,
                           device=torch.device('cpu'))
        for _, _, tok_tp1 in train_triplets
    ]
    pos_weight = compute_pos_weights(all_labels, NUM_CLASSES, device)
    print(f"  pos_weight per class: {pos_weight.tolist()}")

    CLASS_NAMES = ['lane_divider', 'ped_crossing', 'road_segment']

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("\nStarting training...")
    history   = []
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou  = torch.zeros(NUM_CLASSES)
        n_samples  = 0

        np.random.shuffle(train_triplets)

        for i, (tok_tm1, tok_t, tok_tp1) in enumerate(train_triplets):

            # Camera and LiDAR features at t+1 (what we're aligning)
            images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(
                nusc, tok_tp1,
            )
            images           = images.unsqueeze(0).to(device)
            intrinsics       = intrinsics.unsqueeze(0).to(device)
            cam2egos         = cam2egos.unsqueeze(0).to(device)
            img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

            with torch.no_grad():
                cam_bev   = fastbev(images, cam2egos, intrinsics,
                                    img_aug_matrices)['bev_feat']
                raw_pts   = load_lidar_points(nusc, tok_tp1)
                lidar_bev = pointpillars(raw_pts, device)

            # Predicted occupancy: t → t+1 (built from tok_t, not tok_tp1)
            with torch.no_grad():
                pred_bev = build_pred_bev(
                    nusc, tok_tm1, tok_t,
                    args.velocity, yolo_model, args.yolo_conf, device,
                )

            labels = get_bev_map_labels(
                nusc, nusc_maps, tok_tp1,
                bev_extent=BEV_EXTENT, canvas_size=CANVAS_SIZE, device=device,
            ).unsqueeze(0)

            # Channel dropout: randomly zero pred_bev so the model learns
            # to work without it, preventing over-reliance on a channel that
            # may be sparse or differently distributed at val/test time.
            if torch.rand(1).item() < PRED_DROPOUT:
                pred_bev = torch.zeros_like(pred_bev)

            optimizer.zero_grad()
            logits, delta = model(cam_bev, lidar_bev, pred_bev)
            loss = seg_loss(logits, labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                iou = compute_iou(logits, labels)

            epoch_loss += loss.item()
            epoch_iou  += iou
            n_samples  += 1

            if (i + 1) % LOG_EVERY == 0:
                print(f"  Epoch {epoch:3d}  [{i+1:3d}/{len(train_triplets)}]  "
                      f"loss={loss.item():.4f}  "
                      f"IoU=[{iou[0]:.3f}, {iou[1]:.3f}, {iou[2]:.3f}]  "
                      f"delta_max={delta.abs().max().item():.4f}")

        scheduler.step()

        mean_loss    = epoch_loss / n_samples
        mean_iou     = epoch_iou  / n_samples
        mean_iou_val = mean_iou.mean().item()

        print(f"\nEpoch {epoch:3d}/{EPOCHS}  "
              f"loss={mean_loss:.4f}  mIoU={mean_iou_val:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print("  Per-class IoU: "
              + "  ".join(f"{CLASS_NAMES[c]}={mean_iou[c]:.4f}"
                          for c in range(NUM_CLASSES)))

        val_loss, val_miou = None, None

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            print(f"  Running validation ({len(val_triplets)} triplets)...")
            model.eval()
            v_loss = 0.0
            v_iou  = torch.zeros(NUM_CLASSES)
            n_val  = 0

            with torch.no_grad():
                for tok_tm1, tok_t, tok_tp1 in val_triplets:
                    images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(
                        nusc, tok_tp1,
                    )
                    images           = images.unsqueeze(0).to(device)
                    intrinsics       = intrinsics.unsqueeze(0).to(device)
                    cam2egos         = cam2egos.unsqueeze(0).to(device)
                    img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

                    cam_bev   = fastbev(images, cam2egos, intrinsics,
                                        img_aug_matrices)['bev_feat']
                    raw_pts   = load_lidar_points(nusc, tok_tp1)
                    lidar_bev = pointpillars(raw_pts, device)

                    pred_bev = build_pred_bev(
                        nusc, tok_tm1, tok_t,
                        args.velocity, yolo_model, args.yolo_conf, device,
                    )

                    labels = get_bev_map_labels(
                        nusc, nusc_maps, tok_tp1,
                        bev_extent=BEV_EXTENT, canvas_size=CANVAS_SIZE, device=device,
                    ).unsqueeze(0)

                    logits, _ = model(cam_bev, lidar_bev, pred_bev)
                    loss      = seg_loss(logits, labels, pos_weight=pos_weight)
                    v_loss   += loss.item()
                    v_iou    += compute_iou(logits, labels)
                    n_val    += 1

            val_loss = v_loss / n_val
            val_miou = (v_iou / n_val).mean().item()
            val_iou  = v_iou / n_val

            print(f"  Val  loss={val_loss:.4f}  mIoU={val_miou:.4f}")
            print("  Val IoU: "
                  + "  ".join(f"{CLASS_NAMES[c]}={val_iou[c]:.4f}"
                               for c in range(NUM_CLASSES)))

        ckpt = {
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'train_loss': mean_loss,
            'train_miou': mean_iou_val,
            'val_loss':   val_loss,
            'val_miou':   val_miou,
            'velocity':   args.velocity,
            'pred_channels': PRED_CHANNELS,
        }
        torch.save(ckpt, ckpt_last)

        monitor = val_loss if val_loss is not None else mean_loss
        if monitor < best_loss:
            best_loss = monitor
            torch.save(ckpt, ckpt_best)
            print(f"  ✓ New best checkpoint (monitor={best_loss:.4f})")

        history.append({
            'epoch':      epoch,
            'train_loss': mean_loss,
            'train_miou': mean_iou_val,
            'val_loss':   val_loss,
            'val_miou':   val_miou,
            'lr':         scheduler.get_last_lr()[0],
        })

    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Log saved to {log_path}")


if __name__ == '__main__':
    main()
