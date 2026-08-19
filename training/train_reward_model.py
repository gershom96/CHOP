"""Minimal training helpers for the trajectory-conditioned reward model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from models.trajectory_reward_model import bradley_terry_loss


def reward_model_pairwise_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    temperature: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Score winner and loser paths from an uncollapsed raw preference vote.

    Required batch keys: image, preferred_path, rejected_path, intrinsics,
    t_cam_from_base. Optional waypoint masks are preferred_waypoint_valid and
    rejected_waypoint_valid.
    """
    shared = {"intrinsics": batch["intrinsics"], "t_cam_from_base": batch["t_cam_from_base"]}
    # A comparison has one observation and two trajectory hypotheses.  DINO is
    # frozen and image-only, so run it exactly once and reuse its patch map for
    # both candidate-path reward heads.
    image_features = model.encode_image(batch["image"])
    preferred = model(batch["image"], batch["preferred_path"], **shared,
                      waypoint_valid=batch.get("preferred_waypoint_valid"),
                      encoded_image_features=image_features)
    rejected = model(batch["image"], batch["rejected_path"], **shared,
                     waypoint_valid=batch.get("rejected_waypoint_valid"),
                     encoded_image_features=image_features)
    loss, accuracy = bradley_terry_loss(preferred, rejected, temperature)
    return loss, {
        "reward_bt_loss": float(loss.detach()),
        "reward_pair_accuracy": float(accuracy.detach()),
        "reward_margin": float((preferred - rejected).detach().mean()),
    }


def reward_model_grouped_pairwise_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    temperature: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Bradley--Terry step that encodes each frame once for all its pairs."""
    pair_image_index = batch["pair_image_index"]
    cached_features = batch.get("feature_map")
    image_features = cached_features if cached_features is not None else model.encode_image(batch["image"])
    pair_image = None if cached_features is not None else batch["image"].index_select(0, pair_image_index)
    shared = {
        "intrinsics": batch["intrinsics"].index_select(0, pair_image_index),
        "t_cam_from_base": batch["t_cam_from_base"].index_select(0, pair_image_index),
        "encoded_image_features": image_features.index_select(0, pair_image_index),
        "image_size": batch.get("image_size"),
    }
    preferred = model(pair_image, batch["preferred_path"], **shared)
    rejected = model(pair_image, batch["rejected_path"], **shared)
    if not torch.isfinite(preferred).all() or not torch.isfinite(rejected).all():
        raise FloatingPointError("Non-finite reward score; aborting before corrupting Bradley--Terry metrics")
    loss, accuracy = bradley_terry_loss(preferred, rejected, temperature)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Bradley--Terry loss")
    return loss, {
        "reward_bt_loss": float(loss.detach()),
        "reward_pair_accuracy": float(accuracy.detach()),
        "reward_margin": float((preferred - rejected).detach().mean()),
    }
