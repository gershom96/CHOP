import torch
from torch import nn

from models.trajectory_reward_model import TrajectoryAnchorRewardModel, bradley_terry_loss
from training.train_reward_model import reward_model_pairwise_step


def _calibration(batch_size=2):
    # Camera convention for this unit test: camera z points forward, x right,
    # y down; base x forward, base y left, base z up.
    transform = torch.tensor([[0., -1., 0., 0.], [0., 0., -1., .5], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    intrinsics = torch.tensor([[40., 0., 32.], [0., 40., 24.], [0., 0., 1.]])
    return intrinsics.expand(batch_size, -1, -1), transform.expand(batch_size, -1, -1)


def test_anchor_reward_model_projects_and_scores_paths():
    torch.manual_seed(0)
    model = TrajectoryAnchorRewardModel(
        feature_dim=24, hidden_dim=24, num_heads=4, num_layers=1, vision_backbone="cnn"
    )
    image = torch.randn(2, 3, 48, 64)
    path = torch.tensor([[[1., 0.], [2., 0.], [3., .1]], [[1., 0.], [2., .1], [3., .2]]])
    intrinsics, transform = _calibration()
    scores, details = model(image, path, intrinsics, transform, return_details=True)
    assert scores.shape == (2,)
    assert details["anchor_grid"].shape == (2, 3, 2)
    assert details["anchor_valid"].all()


def test_bradley_terry_prefers_larger_scores():
    loss, accuracy = bradley_terry_loss(torch.tensor([2., 3.]), torch.tensor([0., 1.]))
    assert loss < 0.2
    assert accuracy == 1.0


def test_pairwise_step_reuses_one_image_encoding():
    class CountingEncoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.calls = 0

        def forward(self, image):
            self.calls += 1
            return self.encoder(image)

    model = TrajectoryAnchorRewardModel(
        feature_dim=24, hidden_dim=24, num_heads=4, num_layers=1, vision_backbone="cnn"
    )
    model.image_encoder = CountingEncoder(model.image_encoder)
    image = torch.randn(2, 3, 48, 64)
    path = torch.tensor([[[1., 0.], [2., 0.], [3., .1]], [[1., 0.], [2., .1], [3., .2]]])
    intrinsics, transform = _calibration()
    loss, _ = reward_model_pairwise_step(model, {
        "image": image,
        "preferred_path": path,
        "rejected_path": path + torch.tensor([0., .1]),
        "intrinsics": intrinsics,
        "t_cam_from_base": transform,
    })
    assert torch.isfinite(loss)
    assert model.image_encoder.calls == 1
