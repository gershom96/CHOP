#!/usr/bin/env python3
"""Cache frozen DINOv3 feature maps once for fast reward-model training."""

from __future__ import annotations

import argparse
import json
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
            with env.begin() as transaction:
                missing = [index for index, path in enumerate(batch["image_paths"]) if transaction.get(path.encode()) is None]
            if not missing:
                continue
            indices = torch.tensor(missing, dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                images = batch["image"].index_select(0, indices).to(device, non_blocking=True)
                features = model.encode_image(images).half().cpu().contiguous()
            with env.begin(write=True) as transaction:
                for index, feature in zip(missing, features):
                    transaction.put(batch["image_paths"][index].encode(), feature.numpy().tobytes())
                    written += 1
            if written and written % 1024 == 0:
                print(f"cached={written}", flush=True)
    env.sync()
    env.close()
    (args.output / "complete.json").write_text(json.dumps({"written_this_run": written}) + "\n")
    print(f"cached={written} feature maps at {args.output}")


if __name__ == "__main__":
    main()
