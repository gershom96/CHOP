"""Preference-learning losses shared by CHOP training entry points.

The score is deliberately defined from the policy's trajectory prediction, so
no reward model or on-policy rollouts are required.  Given a preferred and a
rejected trajectory for the same observation, the loss is the Bradley--Terry
negative log-likelihood that the preferred trajectory receives the higher
score (lower prediction error).
"""

import torch
import torch.nn.functional as F


def trajectory_ranking_loss(
    prediction: torch.Tensor,
    preferred: torch.Tensor,
    rejected: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Bradley--Terry loss and pairwise accuracy for trajectory pairs.

    ``prediction``, ``preferred``, and ``rejected`` must have matching batch
    and action dimensions.  A trajectory score is its negative mean-squared
    error, hence the loss rewards predictions closer to the human-preferred
    target than to the rejected alternative.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if prediction.shape != preferred.shape or prediction.shape != rejected.shape:
        raise ValueError("prediction, preferred, and rejected must share a shape")

    preferred_error = (prediction - preferred).square().flatten(1).mean(dim=1)
    rejected_error = (prediction - rejected).square().flatten(1).mean(dim=1)
    # score(preferred) - score(rejected) = rejected_error - preferred_error
    loss = F.softplus((preferred_error - rejected_error) / temperature).mean()
    accuracy = (preferred_error < rejected_error).float().mean()
    return loss, accuracy
