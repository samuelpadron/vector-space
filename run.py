"""
run.py — FastBEV4D training entry point.

Trains the temporal-fusion model (BEVDet4D-style, Option A: concat + 1x1 conv)
on top of a pretrained FastBEV backbone.

Usage:
    python run.py
"""

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
NUSCENES_SPLIT  = 'train'           # 'train' | 'mini_train'
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
SAVE_DIR        = Path('./checkpoints/fastbev4d')

NUM_EPOCHS  = 20
LR_BACKBONE = 2e-5   # pretrained weights — keep small
LR_FUSION   = 2e-4   # temporal_fusion is new — can learn faster
GRAD_CLIP   = 35.0
LOG_EVERY   = 10     # steps between loss prints
SAVE_EVERY  = 1      # epochs between checkpoint saves


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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    fusion_ids  = set(id(p) for p in model.temporal_fusion.parameters())
    backbone_ps = [p for p in model.parameters() if id(p) not in fusion_ids]
    fusion_ps   = list(model.temporal_fusion.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_ps, 'lr': LR_BACKBONE},
        {'params': fusion_ps,   'lr': LR_FUSION},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print("\nLoading nuScenes...")
    nusc    = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)
    dataset = NuScenesSequenceDataset(nusc, split=NUSCENES_SPLIT)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # NuScenes object not picklable across workers
    )
    print(f"  {len(dataset)} consecutive-frame pairs")

    criterion = CenterPointLoss(num_classes=10)

    print("\nStarting training...\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            img_curr  = batch['img_curr'].to(device)         # [B, 3, H, W]
            c2e_curr  = batch['cam2ego_curr'].to(device)     # [B, 4, 4]
            intr_curr = batch['intrinsics_curr'].to(device)  # [B, 3, 3]
            img_prev  = batch['img_prev'].to(device)
            c2e_prev  = batch['cam2ego_prev'].to(device)
            intr_prev = batch['intrinsics_prev'].to(device)
            se2       = batch['se2'].to(device)              # [B, 3]
            gt_boxes  = batch['gt_boxes']                    # list[list[dict]]

            # Encode previous frame — detached, as in BEVDet4D
            with torch.no_grad():
                bev_feat_prev, _ = model.encode(img_prev, c2e_prev, intr_prev)
            bev_feat_prev = bev_feat_prev.detach()

            # Forward current frame with temporal fusion
            optimizer.zero_grad()
            outputs = model(
                img_curr, c2e_curr, intr_curr,
                bev_feat_prev=bev_feat_prev,
                se2=se2,
            )

            losses = criterion(outputs['predictions'], gt_boxes)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += losses['loss'].item()

            if step % LOG_EVERY == 0:
                print(
                    f"  Epoch {epoch:02d}  step {step:04d}/{len(loader)}  "
                    f"loss={losses['loss'].item():.4f}  "
                    f"hm={losses['heatmap_loss'].item():.4f}  "
                    f"reg={losses['reg_loss'].item():.4f}"
                )

        scheduler.step()
        avg = epoch_loss / len(loader)
        lr  = scheduler.get_last_lr()[1]   # fusion head LR
        print(f"\nEpoch {epoch:02d} complete — avg loss: {avg:.4f}  lr: {lr:.2e}\n")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = SAVE_DIR / f'epoch_{epoch:02d}.pth'
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'avg_loss':  avg,
            }, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}\n")

    print("Training complete.")


if __name__ == '__main__':
    main()
