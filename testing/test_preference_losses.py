import torch

from training.preference_losses import trajectory_ranking_loss


def test_ranking_loss_rewards_preferred_trajectory():
    preferred = torch.zeros(2, 3, 2)
    rejected = torch.ones(2, 3, 2)
    good_loss, good_accuracy = trajectory_ranking_loss(preferred, preferred, rejected)
    bad_loss, bad_accuracy = trajectory_ranking_loss(rejected, preferred, rejected)
    assert good_loss < bad_loss
    assert good_accuracy == 1.0
    assert bad_accuracy == 0.0
