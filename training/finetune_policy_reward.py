"""GNM/ViNT direct reward optimization from public checkpoints; no SFT loss."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.policy_reward_dataset import PolicyRewardDataset, policy_reward_collate
from training.policy_reward import FrozenPolicyReward
from training.policy_reward_loading import ResumableOrder, legacy_first_epoch_order


def load_policy(path):
    # Original upstream checkpoints pickle their model objects.
    sys.path.insert(
        0,
        str(
            Path(__file__).resolve().parents[1]
            / "policy_sources/visualnav_transformer/train"
        ),
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    policy = saved["model"]
    if isinstance(policy, torch.nn.DataParallel):
        policy = policy.module
    if not isinstance(policy, torch.nn.Module):
        raise TypeError(
            "Expected the original public checkpoint containing a model object"
        )
    if type(policy).__name__ == "ViNT":
        # Rebuild old pickled torch layers under the installed torch version.
        from policy_sources.visualnav_transformer.train.vint_train.models.vint.vint import (
            ViNT,
        )

        layer = policy.decoder.sa_decoder.layers[0]
        fresh = ViNT(
            context_size=policy.context_size,
            len_traj_pred=policy.len_trajectory_pred,
            learn_angle=policy.learn_angle,
            obs_encoding_size=policy.obs_encoding_size,
            late_fusion=policy.late_fusion,
            mha_num_attention_heads=layer.self_attn.num_heads,
            mha_num_attention_layers=len(policy.decoder.sa_decoder.layers),
            mha_ff_dim_factor=layer.linear1.out_features // policy.obs_encoding_size,
        )
        fresh.load_state_dict(policy.state_dict(), strict=True)
        policy = fresh
    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["gnm", "vint"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reward-checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--train-split", required=True)
    p.add_argument("--val-split", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If positive, overrides steps and traverses all training observations each epoch",
    )
    p.add_argument("--monitor-limit", type=int, default=64)
    p.add_argument("--train-limit", type=int, default=512)
    p.add_argument("--val-limit", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--reference-weight", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--policy-image-cache", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--include-uncached",
        action="store_true",
        help="Include all eligible observations; compute missing frozen DINO features online",
    )
    a = p.parse_args()
    train_bags = {r["bag"] for r in json.loads(Path(a.train_split).read_text())}
    val_bags = {r["bag"] for r in json.loads(Path(a.val_split).read_text())}
    if train_bags & val_bags:
        raise ValueError("Training and validation bags must be disjoint")
    if (
        a.reference_weight < 0
        or a.lr <= 0
        or a.steps < 1
        or a.epochs < 0
        or a.eval_every < 1
        or a.monitor_limit < 1
        or a.workers < 0
    ):
        raise ValueError("Invalid optimization settings")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))
    policy = load_policy(a.checkpoint).to(device).eval()
    if a.model not in type(policy).__name__.lower():
        raise ValueError("Checkpoint model does not match requested architecture")
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    objective = FrozenPolicyReward(a.reward_checkpoint, a.reference_weight).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)
    resume = None
    start_step = 0
    if a.resume:
        resume = torch.load(a.resume, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "checkpoint",
            "reward_checkpoint",
            "index",
            "train_split",
            "val_split",
            "batch_size",
            "seed",
            "reference_weight",
            "lr",
            "train_limit",
            "val_limit",
            "include_uncached",
        ):
            if str(resume["config"].get(key)) != str(getattr(a, key)):
                raise ValueError(f"Resume configuration mismatch: {key}")
        policy.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = resume["step"]
        print(f"resuming_step={start_step} checkpoint={a.resume}", flush=True)
    loader_options = {"num_workers": a.workers, "pin_memory": device.type == "cuda"}
    if a.workers:
        loader_options.update(
            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2
        )
    loaders = []
    for split, limit in [(a.train_split, a.train_limit), (a.val_split, a.val_limit)]:
        ds = PolicyRewardDataset(
            a.index,
            split,
            a.image_root,
            a.feature_cache,
            a.calibration,
            limit,
            a.seed,
            include_uncached=a.include_uncached,
            policy_image_cache=a.policy_image_cache,
        )
        sampler = ResumableOrder(len(ds)) if split == a.train_split else None
        loaders.append(
            DataLoader(
                ds,
                batch_size=a.batch_size,
                sampler=sampler,
                **loader_options,
                collate_fn=policy_reward_collate,
            )
        )
        print(f"dataset={split} observations={len(ds)}", flush=True)
    steps_per_epoch = len(loaders[0])
    total_steps = a.epochs * steps_per_epoch if a.epochs else a.steps
    monitor_loader = DataLoader(
        Subset(
            loaders[1].dataset, range(min(a.monitor_limit, len(loaders[1].dataset)))
        ),
        batch_size=a.batch_size,
        collate_fn=policy_reward_collate,
        pin_memory=device.type == "cuda",
    )
    dataset_info = {
        "train_observations": len(loaders[0].dataset),
        "validation_observations": len(loaders[1].dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_coverage": loaders[0].dataset.coverage,
        "validation_coverage": loaders[1].dataset.coverage,
    }
    print(json.dumps(dataset_info), flush=True)
    (a.output / "dataset.json").write_text(json.dumps(dataset_info, indent=2))
    # Capture the same RNG point as the original runner, before its initial
    # monitor and training iterators. No data must be loaded to skip old batches.
    initial_order = legacy_first_epoch_order(
        torch.get_rng_state(), len(loaders[0].dataset)
    )
    current_epoch = start_step // steps_per_epoch
    if resume and "data_order" not in resume and current_epoch != 0:
        raise ValueError(
            "Legacy checkpoints after epoch 1 lack recoverable sampler state"
        )

    def set_order(epoch, skip=0):
        order = loaders[0].sampler
        if resume and resume.get("data_epoch") == epoch:
            order.order = resume["data_order"]
        elif epoch == 0:
            order.order = initial_order
        else:
            order.order = torch.randperm(
                len(loaders[0].dataset),
                generator=torch.Generator().manual_seed(a.seed + epoch),
            )
        order.start = skip * a.batch_size

    set_order(current_epoch, start_step % steps_per_epoch)
    run = None
    if not a.disable_wandb:
        import wandb

        run = wandb.init(
            project="CHOP",
            entity="gershom-university-of-maryland",
            name=f"{a.model}-reward-only-{'full' if not a.train_limit else 'pilot'}",
            config={**vars(a), **dataset_info},
        )

    def step_batch(batch, grad):
        b = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if b["reward_image"] is not None:
            # Match the existing FP16 cache generation. One image encoding is
            # shared by the current-policy and reference-policy reward scores.
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ),
            ):
                features = objective.reward.encode_image(b["reward_image"])
            b["feature_map"][b["missing_features"]] = features.to(
                b["feature_map"].dtype
            )
        # Eval mode disables dropout and BN running-stat drift, but preserves gradients.
        with torch.no_grad():
            _, base = reference(b["obs"], b["goal"])
        with torch.set_grad_enabled(grad):
            _, predicted = policy(b["obs"], b["goal"])
            loss, metrics = objective(
                predicted, base, b["feature_map"], b["intrinsics"], b["transform"]
            )
            if grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 1.0, error_if_nonfinite=True
                )
                if norm == 0:
                    raise RuntimeError("No gradient reached policy")
                optimizer.step()
                metrics["grad_norm"] = norm
        return {k: float(v) for k, v in metrics.items()}, len(base)

    def evaluate(step, full=True):
        total = {}
        count = 0
        for b in loaders[1] if full else monitor_loader:
            metrics, n = step_batch(b, False)
            count += n
            for k, v in metrics.items():
                total[k] = total.get(k, 0) + v * n
        prefix = "val" if full else "monitor"
        result = {f"{prefix}/{k}": v / count for k, v in total.items()}
        record = {"step": step, "observations": count, **result}
        print(json.dumps(record), flush=True)
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if run:
            run.log(result, step=step)
        return result

    def save_checkpoint(step, filename):
        temporary = a.output / (filename + ".tmp")
        torch.save(
            {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": vars(a),
                "data_epoch": current_epoch,
                "data_order": loaders[0].sampler.order,
                "best_objective": best_objective,
            },
            temporary,
        )
        temporary.replace(a.output / filename)

    evaluate(start_step, full=False)
    # Every full evaluation scores the frozen public reference on exactly the
    # same observations. That supplies the full-data before/after comparison
    # without a redundant pre-training traversal of the entire validation set.
    best_objective = resume.get("best_objective", 0.0) if resume else 0.0
    if not resume:
        save_checkpoint(0, "best.pt")
    iterator = iter(loaders[0])
    started = time.monotonic()
    for step in range(start_step + 1, total_steps + 1):
        loading_started = time.monotonic()
        try:
            b = next(iterator)
        except StopIteration:
            current_epoch += 1
            set_order(current_epoch)
            iterator = iter(loaders[0])
            b = next(iterator)
        data_seconds = time.monotonic() - loading_started
        compute_started = time.monotonic()
        metrics, _ = step_batch(b, True)
        metrics["data_wait_seconds"] = data_seconds
        metrics["compute_seconds"] = time.monotonic() - compute_started
        if step % 10 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "epoch": (step - 1) // steps_per_epoch + 1,
                        **metrics,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            if run:
                run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
        full = step % steps_per_epoch == 0 or step == total_steps
        if step == min(50, a.eval_every) or step % a.eval_every == 0 or full:
            save_checkpoint(step, "latest.pt")
            result = evaluate(step, full=full)
            if full and result["val/objective"] < best_objective:
                best_objective = result["val/objective"]
                save_checkpoint(step, "best.pt")
    (a.output / "completed.json").write_text(
        json.dumps(
            {
                "step": total_steps,
                "epochs": a.epochs,
                "best_validation_objective": best_objective,
            }
        )
    )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
