"""Frozen preference reward for deterministic policies; no demonstration loss."""

import torch
from torch import nn

from models.trajectory_reward_model import TrajectoryAnchorRewardModel


def metric_path(actions, scale, points=8):
    """Resample ego XY waypoints by arc length, matching reward preprocessing.

    Heading channels are not height. Segment selection is piecewise constant;
    interpolation and lengths retain derivatives with respect to policy XY.
    """
    xy = actions.float()[..., :2] * scale
    xyz = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
    xyz = torch.cat((torch.zeros_like(xyz[:, :1]), xyz), 1)
    lengths = torch.linalg.vector_norm(xyz[:, 1:] - xyz[:, :-1], dim=-1)
    cumulative = torch.cat((torch.zeros_like(lengths[:, :1]), lengths.cumsum(1)), 1)
    targets = (
        cumulative[:, -1:]
        * torch.linspace(0, 1, points + 1, device=xyz.device)[None, 1:]
    )
    indices = (
        torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=True)
        - 1
    )
    indices = indices.clamp(0, xyz.shape[1] - 2)
    start = xyz.gather(1, indices[..., None].expand(-1, -1, 3))
    end = xyz.gather(1, (indices + 1)[..., None].expand(-1, -1, 3))
    fraction = (targets - cumulative.gather(1, indices)) / lengths.gather(
        1, indices
    ).clamp_min(1e-6)
    return start + fraction[..., None] * (end - start)


class FrozenPolicyReward(nn.Module):
    def __init__(self, checkpoint, reference_weight=1.0):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = state["args"]
        self.reward = TrajectoryAnchorRewardModel(
            dinov3_model_name=cfg["model"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            attention_mode=cfg.get("attention_mode", "deformable"),
        )
        self.reward.load_state_dict(state["model"])
        self.reward.requires_grad_(False).eval()
        self.reference_weight = reference_weight

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(
        self, actions, reference, feature_map, intrinsics, transform, scale=0.38
    ):
        path = metric_path(actions, scale)
        reference_path = metric_path(reference.detach(), scale)
        with torch.autocast(device_type=actions.device.type, enabled=False):
            score = self.reward(
                None,
                path,
                intrinsics,
                transform,
                encoded_image_features=feature_map.float(),
                image_size=(384, 640),
            )
            with torch.no_grad():
                base_score = self.reward(
                    None,
                    reference_path,
                    intrinsics,
                    transform,
                    encoded_image_features=feature_map.float(),
                    image_size=(384, 640),
                )
            # Metric displacement from the original policy, not a target label.
            deviation = (
                (
                    (actions.float()[..., :2] - reference.detach().float()[..., :2])
                    * scale
                )
                .square()
                .mean()
            )
            loss = -(score - base_score).mean() + self.reference_weight * deviation
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy reward objective")
        return loss, {
            "reward": score.mean().detach(),
            "reference_reward": base_score.mean(),
            "reward_gain": (score - base_score).mean().detach(),
            "reference_mse_m2": deviation.detach(),
            "objective": loss.detach(),
        }
