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
