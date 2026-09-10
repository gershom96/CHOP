import torch

from models.trajectory_reward_model import TrajectoryAnchorRewardModel
from training.policy_reward import FrozenPolicyReward, metric_path


def test_policy_collate_keeps_cached_and_uncached_rows_aligned():
    from datasets.policy_reward_dataset import policy_reward_collate

    batch = policy_reward_collate(
        [
            {
                "feature_map": torch.ones(2, 2),
                "reward_image": None,
                "image_path": "cached",
            },
            {
                "feature_map": torch.zeros(2, 2),
                "reward_image": torch.full((3, 2, 2), 7.0),
                "image_path": "missing",
            },
        ]
    )
    assert batch["missing_features"].tolist() == [1]
    assert batch["reward_image"].shape == (1, 3, 2, 2)
    assert batch["image_path"] == ["cached", "missing"]
    assert batch["feature_map"][0].sum() == 4


def test_full_dataset_does_not_filter_on_cache_or_read_path_targets(
    tmp_path, monkeypatch
):
    import json

    import datasets.policy_reward_dataset as data

    class NoScanEnvironment:
        def begin(self):
            raise AssertionError("Full-data eligibility must not scan the cache")

    cache = tmp_path / "cache"
    monkeypatch.setitem(data._FEATURE_ENVS, str(cache.resolve()), NoScanEnvironment())
    monkeypatch.setattr(
        data, "_calibration_for_bag", lambda *args: (torch.eye(3), torch.eye(4))
    )
    index = tmp_path / "index.json"
    split = tmp_path / "split.json"
    index.write_text(
        json.dumps(
            [
                {
                    "bag": "example",
                    "samples": [
                        {"frame_idx": i, "image_path": f"frame-{i}.jpg"}
                        for i in range(30)
                    ],
                }
            ]
        )
    )
    split.write_text('[{"bag": "example"}]')
    dataset = data.PolicyRewardDataset(
        index, split, tmp_path, cache, "unused", include_uncached=True
    )
    assert len(dataset) == 15
    assert (
        dataset.coverage["selected_observations"]
        == dataset.coverage["complete_context_and_goal"]
    )
    assert not dataset.coverage["cache_filter_enabled"]


def test_epoch_runner_visits_entire_dataset_and_saves_completion(tmp_path, monkeypatch):
    import json
    import sys

    import training.finetune_policy_reward as runner

    visits = []

    class GNM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, obs, goal):
            return None, obs * self.weight

    class Dataset:
        def __init__(self, index, split, *args, **kwargs):
            self.training = "train" in str(split)
            self.coverage = {}

        def __len__(self):
            return 3 if self.training else 2

        def __getitem__(self, index):
            if self.training:
                visits.append(index)
            return {
                "obs": torch.tensor([1.0]),
                "goal": torch.tensor([1.0]),
                "feature_map": torch.ones(1),
                "reward_image": None,
                "intrinsics": torch.ones(1),
                "transform": torch.ones(1),
            }

    class Objective(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, predicted, base, *args):
            loss = -(predicted - base).mean()
            return loss, {"objective": loss.detach(), "reward_gain": -loss.detach()}

    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    train.write_text('[{"bag": "train"}]')
    val.write_text('[{"bag": "val"}]')
    output = tmp_path / "output"
    monkeypatch.setattr(runner, "load_policy", lambda _: GNM())
    monkeypatch.setattr(runner, "PolicyRewardDataset", Dataset)
    monkeypatch.setattr(runner, "FrozenPolicyReward", Objective)
    args = [
        "test",
        "--model",
        "gnm",
        "--train-split",
        str(train),
        "--val-split",
        str(val),
        "--output",
        str(output),
        "--epochs",
        "2",
        "--batch-size",
        "2",
        "--eval-every",
        "1",
        "--disable-wandb",
    ]
    for arg in [
        "checkpoint",
        "reward-checkpoint",
        "index",
        "image-root",
        "feature-cache",
        "calibration",
    ]:
        args.extend(["--" + arg, "unused"])
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    assert sorted(visits) == [0, 0, 1, 1, 2, 2]
    assert json.loads((output / "completed.json").read_text())["step"] == 4
    assert (output / "best.pt").is_file()
    saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    visits.clear()
    monkeypatch.setattr(
        sys, "argv", args + ["--resume", str(output / "latest.pt"), "--epochs", "3"]
    )
    runner.main()
    resumed = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert sorted(visits) == [0, 1, 2]
    assert resumed["step"] == 6
    assert resumed["model"]["weight"] > saved["model"]["weight"]
    assert resumed["optimizer"]["state"][0]["step"] == 6


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
def test_scheduler_signal_requests_safe_checkpoint(monkeypatch):
    import signal
    from training.finetune_policy_reward import install_stop_handlers

    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    state = install_stop_handlers()
    assert state["signal"] is None
    handlers[signal.SIGUSR1](signal.SIGUSR1, None)
    assert state["signal"] == "SIGUSR1"
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert state["signal"] == "SIGTERM"
