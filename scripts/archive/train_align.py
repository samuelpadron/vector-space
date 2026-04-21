"""
train_align.py - End-to-end training of AlignNet + SegHead.

Trains the BEVAlignSegNet module (AlignNet + SegHead) using nuScenes BEV
map segmentation as supervision. Only AlignNet and SegHead have learnable parameters.

The segmentation loss propagates gradients through the warp back into
AlignNet, forcing it to learn offsets that bring corresponding scene
elements into spatial agreement — making the alignment semantically
grounded rather than optimising a meaningless cross-modal MSE.

Pipeline
--------
FastBEV (frozen)       → cam_bev  [1, 256, 128, 128]
PointPillars (frozen)  → lidar_bev [1, 256, 128, 128]
BEVAlignSegNet (train) → logits [1, 3, 128, 128]
nuScenes map labels    → labels [1, 3, 128, 128]
BCE loss               → backward -> AlignNet + SegHead update

Usage
-----
    python train_align.py

Outputs
-------
    checkpoints/align_seg_best.pth   - best checkpoint by val loss
    checkpoints/align_seg_last.pth   - last epoch checkpoint
    train_log.json                   - per-epoch loss and IoU history
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

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules import FastBEV, PretrainedPointPillars, load_lidar_points
from data import load_sample, load_checkpoint
from map_labels import load_nusc_maps, get_bev_map_labels, NUM_CLASSES
from seg_align import BEVAlignSegNet, seg_loss, compute_pos_weights


NUSCENES_ROOT   = Path('./data/nuscenes')
FASTBEV_CKPT    = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
PP_CKPT         = Path('./models/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
CKPT_DIR        = Path('./checkpoints')

NUSCENES_VER    = 'v1.0-trainval'
BEV_EXTENT      = 51.2       # metres, half-width of BEV grid
CANVAS_SIZE     = (128, 128) # must match cam_bev spatial dims

# Training
EPOCHS          = 10
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
LOG_EVERY       = 50          # print every N samples
VAL_EVERY       = 5          # run validation every N epochs

# Model
CAM_CHANNELS    = 256
LIDAR_CHANNELS  = 256
MID_CHANNELS    = 128
MAX_OFFSET      = 0.1        # max warp in normalised units (~6.4m)


def compute_iou(logits: torch.Tensor, labels: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
    """
    Per-class IoU from logits and binary labels.
    Returns [NUM_CLASSES] float tensor.
    """
    preds = (logits.sigmoid() > threshold).float()
    inter = (preds * labels).sum(dim=(0, 2, 3))
    union = (preds + labels).clamp(max=1).sum(dim=(0, 2, 3))
    return (inter / (union + 1e-6)).cpu()


def get_sample_tokens(nusc: NuScenes, split: str = 'mini_train') -> list:
    """Return all sample tokens for a given nuScenes split."""
    splits     = create_splits_scenes()
    split_scenes = set(splits[split])
    return [
        s['token'] for s in nusc.sample
        if nusc.get('scene', s['scene_token'])['name'] in split_scenes
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVAlignSegNet')
    parser.add_argument('--no-align', action='store_true',
                        help='Disable AlignNet (baseline: no warp, SegHead only)')
    return parser.parse_args()


def main():
    args = parse_args()
    no_align = args.no_align
    
    run_name = 'baseline_no_align' if no_align else 'align_seg'
    ckpt_best = CKPT_DIR / f'{run_name}_best.pth'
    ckpt_last = CKPT_DIR / f'{run_name}_last.pth'
    log_path  = Path(f'train_log_{run_name}.json')
 
    print(f"Run: {run_name}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Frozen encoders 
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

    print("Creating BEVAlignSegNet (trainable)...")
    model = BEVAlignSegNet(
        camera_channels=CAM_CHANNELS,
        lidar_channels=LIDAR_CHANNELS,
        mid_channels=MID_CHANNELS,
        max_offset=MAX_OFFSET,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10
    )

    print("\nLoading nuScenes...")
    nusc      = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    nusc_maps = load_nusc_maps(str(NUSCENES_ROOT))

    # Use all mini samples for training
    train_tokens = get_sample_tokens(nusc, split='train')
    val_tokens   = get_sample_tokens(nusc, split='val')
    print(f"  Training samples: {len(train_tokens)}")
    print(f"  Validation samples: {len(val_tokens)}")

    print("\nComputing class weights...")
    all_labels = [
        get_bev_map_labels(nusc, nusc_maps, tok,
                           bev_extent=BEV_EXTENT,
                           canvas_size=CANVAS_SIZE,
                           device=torch.device('cpu'))
        for tok in train_tokens
    ]
    pos_weight = compute_pos_weights(all_labels, NUM_CLASSES, device)
    print(f"  pos_weight per class: {pos_weight.tolist()}")
    
    # Training loop
    print("\nStarting training...")
    history     = []
    best_loss   = float('inf')

    CLASS_NAMES = ['lane_divider', 'ped_crossing', 'road_segment']

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou  = torch.zeros(NUM_CLASSES)
        n_samples  = 0

        np.random.shuffle(train_tokens)

        for i, sample_token in enumerate(train_tokens):
            images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(
                nusc, sample_token
            )
            images           = images.unsqueeze(0).to(device)
            intrinsics       = intrinsics.unsqueeze(0).to(device)
            cam2egos         = cam2egos.unsqueeze(0).to(device)
            img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)

            with torch.no_grad():
                cam_bev   = fastbev(images, cam2egos, intrinsics,
                                    img_aug_matrices)['bev_feat']
                raw_pts   = load_lidar_points(nusc, sample_token)
                lidar_bev = pointpillars(raw_pts, device)

            labels = get_bev_map_labels(
                nusc, nusc_maps, sample_token,
                bev_extent=BEV_EXTENT,
                canvas_size=CANVAS_SIZE,
                device=device,
            ).unsqueeze(0)   # [1, 3, 128, 128]

            optimizer.zero_grad()
            logits, delta = model(cam_bev, lidar_bev)
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
                print(f"  Epoch {epoch:3d}  [{i+1:3d}/{len(train_tokens)}]  "
                      f"loss={loss.item():.4f}  "
                      f"IoU=[{iou[0]:.3f}, {iou[1]:.3f}, {iou[2]:.3f}]  "
                      f"delta_max={delta.abs().max().item():.4f}")

        scheduler.step()

        mean_loss = epoch_loss / n_samples
        mean_iou  = epoch_iou  / n_samples
        mean_iou_val = mean_iou.mean().item()

        print(f"\nEpoch {epoch:3d}/{EPOCHS}  "
              f"loss={mean_loss:.4f}  "
              f"mIoU={mean_iou_val:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  Per-class IoU: "
              + "  ".join(f"{CLASS_NAMES[c]}={mean_iou[c]:.4f}"
                          for c in range(NUM_CLASSES)))

        val_loss     = None
        val_miou     = None

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            print(f"  Running validation ({len(val_tokens)} samples)...")
            model.eval()
            v_loss = 0.0
            v_iou  = torch.zeros(NUM_CLASSES)
            n_val  = 0
 
            with torch.no_grad():
                for val_token in val_tokens:
                    images, intrinsics, cam2egos, img_aug_matrices, _ = load_sample(
                        nusc, val_token
                    )
                    images           = images.unsqueeze(0).to(device)
                    intrinsics       = intrinsics.unsqueeze(0).to(device)
                    cam2egos         = cam2egos.unsqueeze(0).to(device)
                    img_aug_matrices = img_aug_matrices.unsqueeze(0).to(device)
 
                    cam_bev   = fastbev(images, cam2egos, intrinsics,
                                        img_aug_matrices)['bev_feat']
                    raw_pts   = load_lidar_points(nusc, val_token)
                    lidar_bev = pointpillars(raw_pts, device)
 
                    labels = get_bev_map_labels(
                        nusc, nusc_maps, val_token,
                        bev_extent=BEV_EXTENT,
                        canvas_size=CANVAS_SIZE,
                        device=device,
                    ).unsqueeze(0)
 
                    logits, _ = model(cam_bev, lidar_bev)
                    loss      = seg_loss(logits, labels, pos_weight=pos_weight)
                    v_loss   += loss.item()
                    v_iou    += compute_iou(logits, labels)
                    n_val    += 1
 
            val_loss = v_loss / n_val
            val_miou = (v_iou / n_val).mean().item()
            val_iou  = v_iou / n_val
 
            print(f"  Val   loss={val_loss:.4f}  mIoU={val_miou:.4f}")
            print(f"  Val IoU: "
                  + "  ".join(f"{CLASS_NAMES[c]}={val_iou[c]:.4f}"
                               for c in range(NUM_CLASSES)))

        ckpt = {
            'epoch':       epoch,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'scheduler':   scheduler.state_dict(),
            'train_loss':        mean_loss,
            'train_miou':        mean_iou_val,
            'val_loss':          val_loss,
            'val_miou':          val_miou,
        }
        torch.save(ckpt, ckpt_last)

        monitor = val_loss if val_loss is not None else mean_loss
        if monitor < best_loss:
            best_loss = mean_loss
            torch.save(ckpt, ckpt_best)
            print(f"  ✓ New best checkpoint (monitor={best_loss:.4f})")

        history.append({
            'epoch':    epoch,
            'train_loss':     mean_loss,
            'train_miou':     mean_iou_val,
            'train_iou':      mean_iou.tolist(),
            'val_loss':       val_loss,
            'val_miou':       val_miou,
            'lr':       scheduler.get_last_lr()[0],
        })

        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

        print()

    print("Training complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {CKPT_DIR}/")
    print(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()