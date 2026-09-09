# Direct preference-reward policy training

This is separate from CHOP SFT. Initialize from the public policy weights;
freeze an identical reference policy and the preference reward. The loss is
`-mean(reward(policy) - reward(reference)) + beta * mean((policy_xy_m - reference_xy_m)^2)`.
The reference score is a detached baseline; it does not affect gradients.
No preferred trajectory, demonstration MSE, distance label, or ranking-target
loss is used. The L2 term is output-space regularization, not a probabilistic KL.

`policy_reward.metric_path` converts normalized XY to meters (SCAND scale 0.38),
prepends the origin, and differentiably arc-length-resamples to eight waypoints,
dropping the origin as in reward-model preprocessing. Heading channels are not Z.
Reward calibration refers to the original full 1280x720 images resized to
640x384, not the center-cropped policy inputs. DINO and the reward network are
frozen; autograd stays enabled through waypoint geometry and visual sampling.

## GNM and ViNT

Use `python -m training.finetune_policy_reward --help`. Both public checkpoint
formats are supported; ViNT is reconstructed from its saved architecture and
strictly loads its state dict to avoid old PyTorch pickle incompatibilities.
The policy stays in eval mode while optimizing its weights so dropout and batch
normalization state changes do not masquerade as policy improvements.

`bash training/run_gnm_reward_pilot.sh` runs the initial 300-update GNM pilot:
512 fixed training observations, 64 fixed validation observations from disjoint
bags, batch 8, LR 1e-6, reference weight 10. This is a pipeline/optimization pilot,
not a full-data result. It uses complete six-image temporal context and a future
image ten frame indices ahead. Only observations present in the frozen DINO
cache are eligible. All candidate points remain present regardless of FOV.
Validation is at update 0 and every 50 updates. W&B uses project CHOP.

The service logs can be followed with:

```bash
journalctl --user -u chop-gnm-reward-pilot -f
```

Checkpoints and validation JSONL are under
`/media/beast-gamma/Media2/CHOP/reward_model/policy_reward/gnm_pilot_v1`.
Finite loss and gradient guards abort corrupt updates. The saved reference
checkpoint path, reward checkpoint, hyperparameters, and subset seed are in
config.json. `latest.pt` includes optimizer state and policy weights.

## OmniVLA

The existing `training/finetune-omnivla.py` has an opt-in `--reward_checkpoint`
mode. It returns predicted actions before computing SFT losses. The same base
VLA with LoRA disabled and separate frozen copies of the public action/pose
heads provides reference actions. Reward mode loads public heads using
`--reward_public_head_step` (210000 for the local public CAST checkpoint).
Use the original checkpoint directory, not the CHOP-fine-tuned directory.

Required options in addition to normal OmniVLA launch settings:

```text
--dataset_config configs/reward_omnivla.yaml
--reward_checkpoint <selected reward best.pt>
--reward_feature_cache <dinov3_feature_cache>
--reward_train_split <splits_v1/train.json>
--reward_val_split <splits_v1/validation.json>
--reward_reference_weight 10
--reward_action_scale 0.38
```

Target-action tokens are replaced by fixed placeholders in reward mode, and
image/cache metadata survives batching. There is no supervised loss even though
the legacy dataset still reads the annotated index. Requires OmniVLA's own
compatible dependency stack; the lightweight reward environment alone does
not include its TensorFlow/Prismatic/PEFT dependencies. The integration is
unit-tested with an adapter/reference fixture; full OmniVLA checkpoint execution
has not yet been validated. GNM and ViNT have passed real-checkpoint backward tests.

## Interpretation

Reward gain on held-out observations is an optimization diagnostic, not proof
of safer navigation. Check policy/reference trajectories, independent human
preferences, and closed-loop metrics before making a behavioral claim.
The old 24-bag test split was already inspected during earlier reward runs;
it cannot retrospectively be described as an untouched final test set.
