# precompute_bev_cache.py
"""
Precomputes frozen FastBEV encoder outputs for all prev-frame images
and saves them to disk. To run once before training.

Run from project root!

Output:
    ./data/cache/bev_prev_train.pth
    ./data/cache/bev_prev_val.pth
"""

import sys
from pathlib import Path

import torch
import h5py
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from modules import FastBEV4D, load_checkpoint
from data import NuScenesSequenceDataset, collate_fn

NUSCENES_ROOT   = Path('./data/nuscenes')
NUSCENES_VER    = 'v1.0-trainval'
CHECKPOINT_PATH = Path('./models/fastbev-r50-cbgs/epoch_20_ema.pth')
CACHE_DIR       = Path('./data/cache')
BATCH_SIZE      = 16

def build_cache(nusc, split, model, device):
    dataset  = NuScenesSequenceDataset(nusc, split=split)
    out_path = CACHE_DIR / f'bev_prev_{split}.h5'

    if out_path.exists():
        print(f"  Cache already exists at {out_path}, skipping.")
        return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=6,
        pin_memory=True,
    )

    # peek at feature shape from first batch
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_feat, _ = model.encode(
            sample_batch['img_prev'].to(device),
            sample_batch['cam2ego_prev'].to(device),
            sample_batch['intrinsics_prev'].to(device),
        )
    feat_shape = sample_feat.shape[1:]  # (C, H, W)

    with h5py.File(out_path, 'w') as f:
        dset = f.create_dataset(
            'feats',
            shape=(len(dataset), *feat_shape),
            dtype='float32',
            chunks=(1, *feat_shape),  # one chunk per sample, optimal for random access
        )

        model.eval()
        with torch.no_grad():
            for batch in loader:
                indices   = batch['idx'].tolist()
                imgs_prev = batch['img_prev'].to(device)
                c2e_prev  = batch['cam2ego_prev'].to(device)
                intr_prev = batch['intrinsics_prev'].to(device)

                feats, _ = model.encode(imgs_prev, c2e_prev, intr_prev)

                for i, idx in enumerate(indices):
                    dset[idx] = feats[i].cpu().numpy()

    print(f"  Saved -> {out_path}")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = model.to(device)

    print("\nLoading nuScenes...")
    nusc = NuScenes(version=NUSCENES_VER, dataroot=str(NUSCENES_ROOT), verbose=False)

    for split in ('train', 'val'):
        build_cache(nusc, split, model, device)

    print("\nDone. Cache ready for training.")


if __name__ == '__main__':
    main()