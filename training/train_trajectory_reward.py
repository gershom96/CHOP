#!/usr/bin/env python3
"""Train the frozen-DINOv3 trajectory-anchor reward model on raw CHOP pairs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.reward_model_dataset import CHOPRewardPreferenceDataset, reward_pair_group_collate
from models.trajectory_reward_model import TrajectoryAnchorRewardModel
from training.train_reward_model import reward_model_grouped_pairwise_step


def _evaluate(model, loader, device, amp_enabled):
    started = time.perf_counter()
    model.eval()
    values = {"loss": 0.0, "accuracy": 0.0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss, metrics = reward_model_grouped_pairwise_step(model, batch)
            n = batch["preferred_path"].shape[0]
            values["loss"] += loss.item() * n
            values["accuracy"] += metrics["reward_pair_accuracy"] * n
            values["count"] += n
    return {
        "loss": values["loss"] / values["count"],
        "accuracy": values["accuracy"] / values["count"],
        "seconds": time.perf_counter() - started,
    }


def _init_wandb(args: argparse.Namespace, device: torch.device) -> Optional[Any]:
    if args.disable_wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb or pass --disable-wandb.") from error
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config={**config, "device": str(device)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evaluation/scand_cameras.json"))
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/chop_reward_dinov3"))
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="LMDB of frozen DINOv3 maps created by cache_dino_features.py")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None,
                        help="Cap held-out pairs; useful only for a fast pipeline smoke test.")
    parser.add_argument("--wandb-project", default="CHOP")
    parser.add_argument("--wandb-entity", default="gershom-university-of-maryland")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-freq", type=int, default=25)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA FP16 mixed precision.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    run = _init_wandb(args, device)
    train_data = CHOPRewardPreferenceDataset(args.pairs, args.image_root, args.calibration, args.train_index, feature_cache=args.feature_cache)
    test_data = CHOPRewardPreferenceDataset(args.pairs, args.image_root, args.calibration, args.test_index, feature_cache=args.feature_cache)
    if args.limit_train:
        train_data.groups = train_data.groups[:args.limit_train]
    if args.limit_test:
        test_data.groups = test_data.groups[:args.limit_test]
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, collate_fn=reward_pair_group_collate)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, collate_fn=reward_pair_group_collate)
    model = TrajectoryAnchorRewardModel(vision_backbone="dinov3", dinov3_model_name=args.model).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best_accuracy = float("-inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss, _ = reward_model_grouped_pairwise_step(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # ``loss`` is averaged over preference pairs, not source images.
            # Cached-feature batches deliberately have ``image=None``.
            pair_count = batch["preferred_path"].shape[0]
            total_loss += loss.item() * pair_count
            seen += pair_count
            global_step += 1
            if run is not None and global_step % args.wandb_log_freq == 0:
                run.log({"train/bt_loss": loss.item(), "train/epoch": epoch}, step=global_step)
        metrics = _evaluate(model, test_loader, device, amp_enabled)
        train_loss = total_loss / seen
        train_seconds = time.perf_counter() - epoch_started - metrics["seconds"]
        print(f"epoch={epoch} train_bt_loss={train_loss:.4f} test_bt_loss={metrics['loss']:.4f} test_pair_accuracy={metrics['accuracy']:.4f} train_pairs_per_second={seen / train_seconds:.1f} train_seconds={train_seconds:.1f} eval_seconds={metrics['seconds']:.1f}")
        if run is not None:
            run.log({
                "train/epoch_bt_loss": train_loss,
                "eval/bt_loss": metrics["loss"],
                "eval/pair_accuracy": metrics["accuracy"],
                "system/train_pairs_per_second": seen / train_seconds,
                "system/train_seconds": train_seconds,
                "system/eval_seconds": metrics["seconds"],
                "epoch": epoch,
            }, step=global_step)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            checkpoint = args.output / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "test_metrics": metrics, "model_name": args.model}, checkpoint)
            if run is not None:
                run.summary["best_pair_accuracy"] = best_accuracy
                run.summary["best_checkpoint"] = str(checkpoint)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
