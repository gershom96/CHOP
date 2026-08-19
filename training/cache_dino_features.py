#!/usr/bin/env python3
"""Cache frozen DINOv3 feature maps once for fast reward-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
import torch
from torch.utils.data import DataLoader

from datasets.reward_model_dataset import CHOPRewardPreferenceDataset, reward_pair_group_collate
from models.trajectory_reward_model import TrajectoryAnchorRewardModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evaluation/scand_cameras.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = CHOPRewardPreferenceDataset(args.pairs, args.image_root, args.calibration)
    loader = DataLoader(data, args.batch_size, shuffle=False, num_workers=args.workers,
                        persistent_workers=args.workers > 0, collate_fn=reward_pair_group_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryAnchorRewardModel().to(device).eval()
    # 72k ViT-S maps at FP16 require roughly 55 GB; leave headroom for LMDB.
    env = lmdb.open(str(args.output), map_size=80 * 1024**3, subdir=True)
    written = 0
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                features = model.encode_image(batch["image"].to(device, non_blocking=True)).half().cpu().contiguous()
            with env.begin(write=True) as transaction:
                for path, feature in zip(batch["image_paths"], features):
                    if transaction.get(path.encode()) is None:
                        transaction.put(path.encode(), feature.numpy().tobytes())
                        written += 1
            if written and written % 1024 == 0:
                print(f"cached={written}", flush=True)
    env.sync()
    env.close()
    print(f"cached={written} feature maps at {args.output}")


if __name__ == "__main__":
    main()
