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
from torch.utils.data import DataLoader

from datasets.policy_reward_dataset import PolicyRewardDataset
from training.policy_reward import FrozenPolicyReward


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
    p.add_argument("--train-limit", type=int, default=512)
    p.add_argument("--val-limit", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--reference-weight", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--disable-wandb", action="store_true")
    a = p.parse_args()
    train_bags = {r["bag"] for r in json.loads(Path(a.train_split).read_text())}
    val_bags = {r["bag"] for r in json.loads(Path(a.val_split).read_text())}
    if train_bags & val_bags:
        raise ValueError("Training and validation bags must be disjoint")
    if a.reference_weight < 0 or a.lr <= 0 or a.steps < 1:
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
    loaders = []
    for split, limit in [(a.train_split, a.train_limit), (a.val_split, a.val_limit)]:
        ds = PolicyRewardDataset(
            a.index, split, a.image_root, a.feature_cache, a.calibration, limit, a.seed
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=a.batch_size,
                shuffle=split == a.train_split,
                num_workers=0,
            )
        )
        print(f"dataset={split} observations={len(ds)}", flush=True)
    run = None
    if not a.disable_wandb:
        import wandb

        run = wandb.init(
            project="CHOP",
            entity="gershom-university-of-maryland",
            name=f"{a.model}-reward-only-pilot",
            config=vars(a),
        )

    def step_batch(batch, grad):
        b = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
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

    def evaluate(step):
        total = {}
        count = 0
        for b in loaders[1]:
            metrics, n = step_batch(b, False)
            count += n
            for k, v in metrics.items():
                total[k] = total.get(k, 0) + v * n
        result = {f"val/{k}": v / count for k, v in total.items()}
        record = {"step": step, **result}
        print(json.dumps(record), flush=True)
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if run:
            run.log(result, step=step)
        return result

    evaluate(0)
    iterator = iter(loaders[0])
    started = time.monotonic()
    for step in range(1, a.steps + 1):
        try:
            b = next(iterator)
        except StopIteration:
            iterator = iter(loaders[0])
            b = next(iterator)
        metrics, _ = step_batch(b, True)
        if step % 10 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        **metrics,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            if run:
                run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": vars(a),
                },
                a.output / "latest.pt",
            )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
