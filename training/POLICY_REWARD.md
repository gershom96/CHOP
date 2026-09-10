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

### Full-data GNM and ViNT runs

`bash training/run_full_policy_reward.sh gnm` (or `vint`) starts from the public
checkpoint, not the pilot or SFT checkpoint. Three complete shuffled epochs use
79,361 training observations in 79 bags; full validation covers 19,113
observations in the separate 20 validation bags. Of 81,842/19,672 indexed frames,
2,481/559 cannot supply the required six-frame context and future goal image.
There is no random subset, cache-availability filter, or trajectory/FOV filter.
Uncached observations use online frozen DINO extraction with the same processor
and FP16 extraction as the existing cache. The final 24 test bags are not used.

The objective remains reward maximization plus public-reference deviation only,
with LR 1e-6, reference weight 10, batch 8, and 9,921 updates per epoch (29,763
total). A fixed 64-observation monitor runs before training, at update 50, and
every 200 updates. These `monitor/` metrics are not full-validation results.
Full `val/` metrics run after every epoch, including a frozen-public-reference
score on every same observation to provide a paired before/after comparison.
All metrics are sample-weighted, including the partial last batch.

Outputs are `policy_reward/gnm_full_v1` and `policy_reward/vint_full_v1` below the
reward-model data directory. `dataset.json` records exact coverage, `latest.pt`
is saved atomically before evaluation, and `best.pt` minimizes the full-validation
regularized objective (including the original public policy as candidate zero).
`completed.json` is written only after all epochs and final evaluation finish.
Numerical errors stop the run; there is no automatic restart. The launcher
resumes `latest.pt` when present, including optimizer state and sample order.
W&B logs both runs under CHOP. Follow both service logs with:

```bash
journalctl --user -u chop-gnm-reward-full -u chop-vint-reward-full -f
```

Improved learned reward is evidence of optimizing this objective, not independent
proof of safer or more successful navigation. Compare downstream navigation and
human preferences separately before claiming the method improves navigation.

Initial live check (2026-09-09, update 50, **64-observation monitor only**):

| Model | Reward gain over public policy | RMS coordinate deviation |
| --- | ---: | ---: |
| GNM | +0.019601 | 3.54 mm |
| ViNT | +0.046871 | 5.58 mm |

Both regularized monitor objectives improved from zero and both services were
active with finite gradients. This is not a completed epoch or full-validation
result. W&B run IDs: GNM `a117yo1k`, ViNT `8wbna4ch`.

### I/O repair, September 9

Both models resume from their update-3,000 checkpoints. `pre_io_fix.pt` preserves
each pre-repair checkpoint. Uncheckpointed updates after 3,000 are rolled back;
the public reference, reward, LR, batch size, total update budget, and observation
coverage are unchanged. Resumed runs create new W&B runs under CHOP so rolled-back
steps are not silently discarded by W&B's monotonically increasing step rule.

Two spawn-mode workers per training loader prefetch batches while the GPU works.
Workers open their own read-only LMDB handle; CUDA state and parent LMDB handles
are never inherited. Pinned batches use nonblocking device transfers. Training
logs separate `data_wait_seconds` and `compute_seconds` (the latter includes
device transfer, optional DINO extraction, policy/reference/reward and backward).

Both models share `/home/beast-gamma/.cache/chop-policy-images-v1` on the SSD.
This lazy, sharded cache stores the exact PIL-resized 85x64 uint8 pixels before
ImageNet normalization, not approximate image features or lossy JPEGs. Tests and
a real-data SHA256 comparison prove bitwise-identical policy inputs. Entries are
atomically installed and versioned by preprocessing, source root and relative
path; source images are assumed immutable. The expected full cache is about
1.7 GB plus filesystem overhead. Writes stop when SSD free space falls below
3 GiB; uncached data still loads normally. Raw source images and DINO maps stay
untouched. Cold/uncached DINO still incurs source-disk I/O, so cold full-data
training will not attain the warmed-cache benchmark rate immediately.

Benchmark: `python -m testing.benchmark_policy_io --cache <SSD-cache>` read the
same 128 observations in 16 batches, with identical policy-input hashes:

- Original serial pipeline: 56.90 seconds total; 3.319 s/batch after first batch.
- Two workers, cache being populated: 29.36 seconds; 1.199 s/batch after first.
- Two workers, warm SSD cache: 11.30 seconds; 0.112 s/batch after first.

Totals include process startup/shutdown. Sequential benchmark variants share OS
cache warming; these are loader measurements, not promised full-training speedups.
Use live timing logs for end-to-end performance.

Resume now stores the explicit epoch permutation and skips indices without
reading completed batches. For the old first-epoch checkpoints, reconstruct the
original PyTorch sampler seed draws; `testing/check_legacy_policy_order.py`
verified both reconstructed step-10 frozen-reference scores against the original
logs exactly (GNM 5.471641540527344; ViNT 5.744907379150391). Legacy checkpoints
after the first epoch fail closed because their sampler state is unavailable.
Future epochs use explicit seeded permutations. Unit tests verify complete epoch
coverage and optimizer-state continuation on resume.

Live cold-cache follow-up: the first 30 resumed updates still averaged roughly
11 seconds/update, despite only 0.08–0.26 seconds of measured compute in logged
batches. Thus the warm loader benchmark must not be claimed as an immediate
end-to-end speedup. Rebuilding the whole small policy-input cache once is the
next stage of the repair, rather than relying on slow random-access warming.

`training/warm_full_policy_images.sh` suspends both training service process
groups in place (no additional checkpoint rollback), warms the SSD cache in
bag/time order, then sends SIGCONT on success or failure via an EXIT trap.
The bounded four-thread warm-up test processed 1,024 images in 46.11 seconds
(22.21 images/second). The full one-time preparation is therefore approximately
an hour to 80 minutes, subject to source-disk speed and existing cache hits.
The `chop-policy-image-warmup` service runs this stage. A separate
`chop-policy-io-resume-safety.timer` sends SIGCONT after two hours as a backstop.
Training is suspended, not advancing, during this stage. Check its progress:

```bash
journalctl --user -u chop-policy-image-warmup -f
```

Resumed W&B runs: GNM `ztbj1vqo`, ViNT `2qh5yzcr`. Their step-3,000 monitor
metrics match the old checkpoints exactly. Raw images and reward-feature caches
are not changed. End-to-end warmed training speed remains to be measured once
the full cache-preparation stage completes.

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
