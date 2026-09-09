import torch

from models.trajectory_reward_model import TrajectoryAnchorRewardModel
from training.policy_reward import FrozenPolicyReward, metric_path


def test_metric_resampling_and_heading_not_height():
    actions = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0]]], requires_grad=True
    )
    path = metric_path(actions, 0.5)
    torch.testing.assert_close(path[0, :, 0], torch.arange(1, 9) / 8)
    assert not path[..., 2].any()
    path.sum().backward()
    assert torch.isfinite(actions.grad).all()
    assert not actions.grad[..., 2:].any()


def test_reward_only_gradient_reaches_policy_not_reward_or_reference():
    torch.manual_seed(1)
    objective = object.__new__(FrozenPolicyReward)
    torch.nn.Module.__init__(objective)
    objective.reward = (
        TrajectoryAnchorRewardModel(
            vision_backbone="cnn",
            feature_dim=192,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
        )
        .requires_grad_(False)
        .eval()
    )
    objective.reference_weight = 10.0
    policy = torch.nn.Linear(2, 16)
    actions = policy(torch.ones(2, 2)).reshape(2, 8, 2)
    reference = actions.detach().clone().requires_grad_(True)
    k = torch.eye(3).repeat(2, 1, 1)
    t = torch.eye(4).repeat(2, 1, 1)
    t[:, 2, 3] = 1
    loss, _ = objective(actions, reference, torch.randn(2, 192, 24, 40), k, t)
    loss.backward()
    assert torch.isfinite(policy.weight.grad).all() and policy.weight.grad.norm() > 0
    assert reference.grad is None
    assert all(p.grad is None for p in objective.reward.parameters())


def test_omni_reference_disables_adapter_and_avoids_supervised_forward(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import training.omnivla_policy_reward as integration

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.ones(1) * 0.1)
            self.enabled = True

        @contextmanager
        def disable_adapter(self):
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = True

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return self.module(x)

    class Objective(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, p, r, *args):
            assert not r.requires_grad
            return -p.mean() + (p - r).square().mean(), {}

    monkeypatch.setattr(integration, "FrozenPolicyReward", Objective)
    cfg = SimpleNamespace(
        use_lora=True,
        reward_checkpoint="",
        reward_reference_weight=1,
        reward_feature_cache="/tmp/unused",
        reward_calibration="/tmp/unused",
        reward_action_scale=0.38,
    )
    model = Wrapper(Base())
    head = Wrapper(torch.nn.Linear(1, 1))
    pose = Wrapper(torch.nn.Linear(1, 1))
    step = integration.OmniRewardStep(cfg, model, head, pose, "cpu")
    step.cache._cached_feature = lambda _: torch.zeros(384, 24, 40)
    step.calibrations = {"bag": (torch.eye(3), torch.eye(4))}

    def forward(vla, head, pose, batch, return_actions=False):
        assert return_actions, "Must bypass SFT objective"
        vla = vla.module if hasattr(vla, "module") else vla
        x = vla.base + (vla.adapter if vla.enabled else 0)
        return head.module(pose(x)).reshape(1, 1, 1).expand(1, 8, 2)

    loss, _ = step(
        forward,
        model,
        head,
        pose,
        {"reward_image_path": ["frame"], "reward_bag": ["bag"]},
    )
    loss.backward()
    assert model.module.enabled and model.module.adapter.grad is not None
    assert model.module.base.grad is None
    assert all(p.grad is None for p in step.head.module.parameters())
