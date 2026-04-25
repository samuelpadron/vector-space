"""
run.py — FastBEV4D training entry point.

Trains the temporal-fusion model (BEVDet4D-style, Option A: concat + 1x1 conv)
on top of a pretrained FastBEV backbone.

Usage:
    python run.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from modules import FastBEV4D, load_checkpoint, CenterPointLoss
from data import NuScenesSequenceDataset, collate_fn


NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
SAVE_DIR        = Path('./checkpoints/fastbev4d')

NUM_EPOCHS  = 20
LR_FUSION   = 2e-4   # temporal_fusion is new — can learn faster
GRAD_CLIP   = 35.0
LOG_EVERY   = 10     # steps between loss prints
VAL_EVERY   = 5      # epochs between validation runs


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
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.temporal_fusion.parameters())
    print(f"  Parameters: {total:,} total, {trainable:,} trainable (fusion only)")

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.temporal_fusion.parameters():
        p.requires_grad_(True)

    fusion_ps = list(model.temporal_fusion.parameters())
    optimizer = torch.optim.AdamW(fusion_ps, lr=LR_FUSION, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print("\nLoading nuScenes...")
    nusc       = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    train_set  = NuScenesSequenceDataset(nusc, split='train')
    val_set    = NuScenesSequenceDataset(nusc, split='val')
    loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # NuScenes object not picklable across workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"  {len(train_set)} train pairs, {len(val_set)} val pairs")

    criterion = CenterPointLoss(num_classes=10)

    def run_epoch(loader, train=True):
        if train:
            model.eval()
            model.temporal_fusion.train()
        else:
            model.eval()

        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for step, batch in enumerate(loader):
                img_curr  = batch['img_curr'].to(device)
                c2e_curr  = batch['cam2ego_curr'].to(device)
                intr_curr = batch['intrinsics_curr'].to(device)
                img_prev  = batch['img_prev'].to(device)
                c2e_prev  = batch['cam2ego_prev'].to(device)
                intr_prev = batch['intrinsics_prev'].to(device)
                se2       = batch['se2'].to(device)
                gt_boxes  = batch['gt_boxes']

                with torch.no_grad():
                    bev_feat_prev, _ = model.encode(img_prev, c2e_prev, intr_prev)
                bev_feat_prev = bev_feat_prev.detach()

                if train:
                    optimizer.zero_grad()

                outputs = model(
                    img_curr, c2e_curr, intr_curr,
                    bev_feat_prev=bev_feat_prev,
                    se2=se2,
                )
                losses = criterion(outputs['predictions'], gt_boxes)

                if train:
                    losses['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

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
        lr = scheduler.get_last_lr()[0]
        print(f"  train loss: {train_loss:.4f}  lr: {lr:.2e}")

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
            'lr':         lr,
        })
        with open(SAVE_DIR / 'log.json', 'w') as f:
            json.dump(history, f, indent=2)
        print()

    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
