"""OmniVLA reward-only forward integration and metadata-preserving collation."""

import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.reward_model_dataset import (
    CHOPRewardPreferenceDataset,
    _calibration_for_bag,
)
from training.policy_reward import FrozenPolicyReward


class RewardCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, instances):
        output = self.base(instances)
        for key in ("reward_image_path", "reward_bag"):
            output[key] = [item[key] for item in instances]
        return output


class OmniRewardStep:
    def __init__(self, cfg, vla, head, pose, device):
        if not cfg.use_lora:
            raise ValueError(
                "Reward-only OmniVLA currently requires LoRA for a frozen base reference"
            )
        self.objective = FrozenPolicyReward(
            cfg.reward_checkpoint, cfg.reward_reference_weight
        ).to(device)
        self.head = SimpleNamespace(
            module=copy.deepcopy(head.module).requires_grad_(False).eval()
        )
        self.pose = copy.deepcopy(pose.module).requires_grad_(False).eval()
        self.cache = object.__new__(CHOPRewardPreferenceDataset)
        self.cache.feature_cache = Path(cfg.reward_feature_cache)
        self.cache._feature_env = None
        self.cache.image_size = (384, 640)
        self.calibration_path = Path(cfg.reward_calibration)
        self.calibrations = {}
        self.scale = cfg.reward_action_scale

    def __call__(self, forward, vla, action_head, pose_projector, batch, **kwargs):
        vla.eval()
        action_head.eval()
        pose_projector.eval()
        # Same frozen base weights, with adapters disabled and original heads.
        with torch.no_grad(), vla.module.disable_adapter():
            reference = forward(
                vla.module, self.head, self.pose, batch, return_actions=True, **kwargs
            )
        prediction = forward(
            vla, action_head, pose_projector, batch, return_actions=True, **kwargs
        )
        features = torch.stack(
            [self.cache._cached_feature(p) for p in batch["reward_image_path"]]
        ).to(prediction.device)
        for name in set(batch["reward_bag"]):
            if name not in self.calibrations:
                k, t = _calibration_for_bag(self.calibration_path, name)
                k[0] *= 640 / 1280
                k[1] *= 384 / 720
                self.calibrations[name] = (k, t)
        k = torch.stack([self.calibrations[n][0] for n in batch["reward_bag"]]).to(
            prediction.device
        )
        t = torch.stack([self.calibrations[n][1] for n in batch["reward_bag"]]).to(
            prediction.device
        )
        loss, metrics = self.objective(
            prediction, reference, features, k, t, self.scale
        )
        return loss, {
            "loss_value": float(loss.detach()),
            **{key: float(v) for key, v in metrics.items()},
        }
