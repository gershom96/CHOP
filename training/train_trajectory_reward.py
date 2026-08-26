#!/usr/bin/env python3
"""Train the frozen-DINOv3 trajectory-anchor reward model on raw CHOP pairs."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from datasets.reward_model_dataset import CHOPRewardPreferenceDataset, reward_pair_group_collate
from models.trajectory_reward_model import TrajectoryAnchorRewardModel
from training.train_reward_model import reward_model_grouped_pairwise_step


def _evaluate(model, loader, device, amp_enabled, score_regularization=0.0):
    started = time.perf_counter()
    model.eval()
    values = {"loss": 0.0, "accuracy": 0.0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss, metrics = reward_model_grouped_pairwise_step(
                    model, batch, score_regularization=score_regularization,
                )
            n = batch["preferred_path"].shape[0]
            values["loss"] += metrics["reward_bt_loss"] * n
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=1.0,
                        help="Final / peak LR for cosine decay; 1 disables decay.")
    parser.add_argument("--score-regularization", type=float, default=0.0,
                        help="Coefficient on mean squared trajectory reward scores.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many non-improving validation epochs; 0 disables it.")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-mode", choices=("deformable", "global"), default="deformable")
    # The grouped raw-preference index is large.  Forked workers duplicate enough
    # Python metadata to trigger paging, which is much slower than synchronous
    # cached-feature reads.
    parser.add_argument("--workers", type=int, default=0)
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
    model = TrajectoryAnchorRewardModel(
        vision_backbone="dinov3", dinov3_model_name=args.model,
        hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        attention_mode=args.attention_mode,
    ).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                      lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    if args.warmup_steps < 0 or args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps must be non-negative and smaller than total training steps")

    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return float(step + 1) / args.warmup_steps
        if args.min_lr_scale == 1.0:
            return 1.0
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return args.min_lr_scale + (1 - args.min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_factor)
    args.output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_bt_loss, total_score_penalty, seen = 0.0, 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss, metrics = reward_model_grouped_pairwise_step(
                    model, batch, score_regularization=args.score_regularization,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # ``loss`` is averaged over preference pairs, not source images.
            # Cached-feature batches deliberately have ``image=None``.
            pair_count = batch["preferred_path"].shape[0]
            total_bt_loss += metrics["reward_bt_loss"] * pair_count
            total_score_penalty += metrics["reward_score_penalty"] * pair_count
            seen += pair_count
            global_step += 1
            if run is not None and global_step % args.wandb_log_freq == 0:
                run.log({
                    "train/bt_loss": metrics["reward_bt_loss"],
                    "train/score_penalty": metrics["reward_score_penalty"],
                    "train/lr": scheduler.get_last_lr()[0], "train/epoch": epoch,
                }, step=global_step)
        metrics = _evaluate(model, test_loader, device, amp_enabled, args.score_regularization)
        train_loss = total_bt_loss / seen
        train_seconds = time.perf_counter() - epoch_started - metrics["seconds"]
        print(f"epoch={epoch} train_bt_loss={train_loss:.4f} val_bt_loss={metrics['loss']:.4f} val_pair_accuracy={metrics['accuracy']:.4f} train_pairs_per_second={seen / train_seconds:.1f} train_seconds={train_seconds:.1f} eval_seconds={metrics['seconds']:.1f}", flush=True)
        if run is not None:
            run.log({
                "train/epoch_bt_loss": train_loss,
                "train/epoch_score_penalty": total_score_penalty / seen,
                "eval/bt_loss": metrics["loss"],
                "eval/pair_accuracy": metrics["accuracy"],
                "system/train_pairs_per_second": seen / train_seconds,
                "system/train_seconds": train_seconds,
                "system/eval_seconds": metrics["seconds"],
                "epoch": epoch,
            }, step=global_step)
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            stale_epochs = 0
            checkpoint = args.output / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": metrics, "model_name": args.model, "args": vars(args)}, checkpoint)
            if run is not None:
                run.summary["best_val_bt_loss"] = best_val_loss
                run.summary["best_checkpoint"] = str(checkpoint)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_val_bt_loss={best_val_loss:.4f}", flush=True)
            break
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
