#!/usr/bin/env python3
"""Evaluate a selected trajectory reward checkpoint on one bag-disjoint split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.reward_model_dataset import CHOPRewardPreferenceDataset, reward_pair_group_collate
from models.trajectory_reward_model import TrajectoryAnchorRewardModel
from training.train_trajectory_reward import _evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CHOPRewardPreferenceDataset(
        args.pairs, args.image_root, args.calibration, args.index, feature_cache=args.feature_cache,
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=reward_pair_group_collate)
    model = TrajectoryAnchorRewardModel(
        vision_backbone="dinov3", dinov3_model_name=config["model"],
        hidden_dim=config["hidden_dim"], num_heads=config["num_heads"], num_layers=config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = _evaluate(model, loader, device, amp_enabled=device.type == "cuda")
    result = {"checkpoint": str(args.checkpoint), "split_index": str(args.index), **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
