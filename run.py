"""
run.py — FastBEV4D training entry point.

Trains the temporal-fusion model (BEVDet4D-style, concat + 1x1 conv)
on top of a pretrained FastBEV backbone.

Tensor conventions (monocam, N=1)
----------------------------------
imgs        : [B, 1, 3, H, W]
cam2ego     : [B, 1, 4, 4]
intrinsics  : [B, 1, 3, 3]
"""

import json
import sys
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules import FastBEV4D, load_checkpoint, CenterPointLoss
from data import NuScenesSequenceDataset, collate_fn


NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
SAVE_DIR        = Path('./checkpoints/fastbev4d')

NUM_EPOCHS  = 40
LR_FUSION   = 1e-3
GRAD_CLIP   = 5.0
LOG_EVERY   = 10
VAL_EVERY   = 5


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    print("\nBuilding FastBEV4D...")
    model = FastBEV4D(
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
        print(f"  Warning: checkpoint not found at {CHECKPOINT_PATH}, training from scratch")

    model = model.to(device)
    scaler = GradScaler()

    for param in model.parameters():
        param.requires_grad = True

    n_total      = sum(p.numel() for p in model.parameters())
    n_fusion     = sum(p.numel() for p in model.temporal_fusion.parameters())
    n_head       = sum(p.numel() for p in model.pts_bbox_head.parameters())
    n_backbone   = sum(p.numel() for p in model.img_backbone.parameters())
    n_neck       = sum(p.numel() for p in model.img_neck.parameters())
    n_view_trans = sum(p.numel() for p in model.img_view_transformer.parameters())
    n_bev_bb     = sum(p.numel() for p in model.img_bev_encoder_backbone.parameters())
    n_bev_neck   = sum(p.numel() for p in model.img_bev_encoder_neck.parameters())
    print(f"  Parameters: {n_total:,} total")
    print(f"  (fusion={n_fusion:,}, head={n_head:,}, backbone={n_backbone:,})")

    LR_PRETRAINED = LR_FUSION * 0.01   # 1e-5, slow adaptation for pretrained components
    LR_HEAD       = LR_FUSION * 0.1    # 1e-4, medium for head

    optimizer = torch.optim.AdamW([
        {'params': model.img_backbone.parameters(),             'lr': LR_PRETRAINED},
        {'params': model.img_neck.parameters(),                 'lr': LR_PRETRAINED},
        {'params': model.img_view_transformer.parameters(),     'lr': LR_PRETRAINED},
        {'params': model.img_bev_encoder_backbone.parameters(), 'lr': LR_PRETRAINED},
        {'params': model.img_bev_encoder_neck.parameters(),     'lr': LR_PRETRAINED},
        {'params': model.temporal_fusion.parameters(),          'lr': LR_FUSION},
        {'params': model.pts_bbox_head.parameters(),            'lr': LR_HEAD},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    print("\nLoading nuScenes...")
    nusc      = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    train_set = NuScenesSequenceDataset(nusc, split='train')
    val_set   = NuScenesSequenceDataset(nusc, split='val')

    loader = DataLoader(
        train_set, batch_size=16, shuffle=True,
        collate_fn=collate_fn, num_workers=6,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=6,
        pin_memory=True, persistent_workers=True
    )
    print(f"  {len(train_set)} train pairs, {len(val_set)} val pairs")

    criterion = CenterPointLoss(num_classes=10)

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for step, batch in enumerate(loader):
                imgs_curr  = batch['img_curr'].to(device)       # [B, 1, 3, H, W]
                c2e_curr   = batch['cam2ego_curr'].to(device)   # [B, 1, 4, 4]
                intr_curr  = batch['intrinsics_curr'].to(device)# [B, 1, 3, 3]
                imgs_prev  = batch['img_prev'].to(device)       # [B, 1, 3, H, W]
                c2e_prev   = batch['cam2ego_prev'].to(device)   # [B, 1, 4, 4]
                intr_prev  = batch['intrinsics_prev'].to(device)# [B, 1, 3, 3]
                se2        = batch['se2'].to(device)            # [B, 3]
                gt_boxes   = batch['gt_boxes']

                with torch.no_grad():
                    with autocast(device_type=device.type):
                        img_feats_prev = model.extract_img_feat(imgs_prev)
                        bev_feat_prev, _ = model.img_view_transformer(img_feats_prev, c2e_prev, intr_prev)
                bev_feat_prev = bev_feat_prev.detach()

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=device.type):
                    outputs = model(
                        imgs_curr, c2e_curr, intr_curr,
                        bev_feat_prev=bev_feat_prev,
                        se2=se2,
                    )
                    losses = criterion(outputs['predictions'], gt_boxes)

                if train:
                    scaler.scale(losses['loss']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    if step % LOG_EVERY == 0:
                        print(
                            f"    step {step:04d}/{len(loader)}  "
                            f"loss={losses['loss'].item():.4f}  "
                            f"hm={losses['heatmap_loss'].item():.4f}  "
                            f"reg={losses['reg_loss'].item():.4f}"
                        )

                total_loss += losses['loss'].item()

        return total_loss / len(loader)

    print("\nStarting training...\n")
    best_loss = float('inf')
    history   = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS - 1}")
        train_loss = run_epoch(loader, train=True)
        scheduler.step()
        lr_fusion = optimizer.param_groups[0]['lr']
        print(f"  train loss: {train_loss:.4f}  lr={lr_fusion:.2e}")

        val_loss = None
        if epoch % VAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Running validation ({len(val_set)} pairs)...")
            val_loss = run_epoch(val_loader, train=False)
            print(f"  val   loss: {val_loss:.4f}")

        ckpt = {
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss':   val_loss,
        }
        torch.save(ckpt, SAVE_DIR / 'last.pth')

        monitor = val_loss if val_loss is not None else train_loss
        if monitor < best_loss:
            best_loss = monitor
            torch.save(ckpt, SAVE_DIR / 'best.pth')
            print(f"  -> new best (loss={best_loss:.4f}), saved best.pth")

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         lr_fusion,
        })
        with open(SAVE_DIR / 'log.json', 'w') as f:
            json.dump(history, f, indent=2)
        print()

    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()