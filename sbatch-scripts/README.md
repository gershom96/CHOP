# Cluster reward-only policy training

These jobs train **GNM and ViNT policies using the frozen learned reward**, not
the Bradley–Terry reward model itself. No SFT loss or CHOP-finetuned initialization
is used. Defaults match the completed runs: 3 epochs, all eligible observations,
batch 8, LR 1e-6, reference weight 10, and full validation after each epoch.

Defaults were verified over SSH against `/fs/nexus-scratch/gershom/CHOP` on
`nexusgamma` on September 10, 2026. Account/partition `gamma`, QoS `huge-long`,
and GPU types match that checkout's existing ignored `sbatch-scripts` files:
L40S for GNM and RTX A5000 for ViNT. SCAND's path matches CHOP's YAML configs.
Unlike the old SFT scripts, these single-GPU trainers request one GPU and one
task. Original SFT launchers are preserved; separate reward launchers are exposed
as `sbatch-scripts/finetune-gnm-reward.slurm` and `finetune-vint-reward.slurm`.
The launchers, shared `run_policy_reward.sh`, and documentation live directly in
`sbatch-scripts/`; no symlinks or separate `slurm/` directory are needed.
Git ignores the older cluster-specific scripts, but not these reward launchers,
their shared script, or this documentation.

## Submit from the cluster's CHOP checkout

```bash
cd /fs/nexus-scratch/gershom/CHOP

sbatch sbatch-scripts/finetune-gnm-reward.slurm
sbatch sbatch-scripts/finetune-vint-reward.slurm
```

Each submission requests **one GPU**, six CPUs and 48 GB RAM. Time limits match
the old launchers: 43 hours for GNM and 47 hours for ViNT. The trainer
is single-GPU, so requesting multiple GPUs does not accelerate one job. Slurm
assigns visible devices; the launchers do not overwrite `CUDA_VISIBLE_DEVICES`.
Override scheduler resources at submission if needed, e.g.
`sbatch --gres=gpu:rtxa5000:1 sbatch-scripts/finetune-gnm-reward.slurm`.
Logs are `slurm-chop-gnm-reward-<jobid>.out/.err` and the analogous ViNT files in
the submission directory; no pre-existing logs directory is required.

## Expected files and path overrides

`CHOP_PROJECT_ROOT` defaults to `SLURM_SUBMIT_DIR`; run `sbatch` from the checkout
or set it explicitly. The following defaults are relative to that checkout:

| Variable | Default |
| --- | --- |
| `CHOP_PYTHON` | `.venv-reward/bin/python` |
| `CHOP_DATA_ROOT` | `data` |
| `CHOP_REWARD_ROOT` | `$CHOP_DATA_ROOT/reward_model` |
| `CHOP_INDEX` | `$CHOP_DATA_ROOT/lora-data/train.json` |
| `CHOP_TRAIN_SPLIT` | `$CHOP_REWARD_ROOT/splits_v1/train.json` |
| `CHOP_VAL_SPLIT` | `$CHOP_REWARD_ROOT/splits_v1/validation.json` |
| `CHOP_IMAGE_ROOT` | `/fs/gamma-datasets/SCAND/images` |
| `CHOP_PUBLIC_CHECKPOINT` | `weights/gnm.pth` or `weights/vint.pth` |
| `CHOP_REWARD_CHECKPOINT` | `weights/trajectory_reward/compact_lr1e4_no_reg/best.pt` |
| `HF_HOME` | `/gammascratch/gershom/CHOP/huggingface` |
| `CHOP_FEATURE_CACHE` | `/gammascratch/gershom/CHOP/reward_model/dinov3_feature_cache` |
| `CHOP_CALIBRATION` | `evaluation/scand_cameras.json` |
| `CHOP_OUTPUT` | `$CHOP_REWARD_ROOT/policy_reward/<model>_cluster_reward_v1` |

Copy the reward checkpoint, bag splits, calibration, public policy checkpoints,
and frozen DINO LMDB to these cluster locations, or override each variable.
The large reward artifacts and calibration are not guaranteed to be present in
a fresh Git checkout. Copy LMDB `data.mdb` only from a quiescent/completed cache;
do not copy a database while another job is modifying it. Raw pair labels are
not inputs to this policy-optimization stage.

The verified CHOP checkout already has `data/lora-data/train.json`, `test.json`,
and public `weights/gnm.pth` / `weights/vint.pth`. All 101,514 images referenced
by the training index were verified present on September 10, 2026. Reward model
files belong alongside public policy weights; bag splits stay under `data/`.
The DINO cache is deliberately outside `/fs/nexus-scratch`: at inspection that
allocation had only about 33 GiB free, versus about 74 GiB on gamma scratch.

## UV environment and model access

Use a separate UV environment, **not** the root project's OmniVLA-pinned
`pyproject.toml` or Conda environment:

```bash
export UV_CACHE_DIR=/gammascratch/gershom/CHOP/uv-cache
export UV_PYTHON_INSTALL_DIR=/gammascratch/gershom/CHOP/uv-python
UV_BIN=/gammascratch/gershom/CHOP/tools/uv
"$UV_BIN" venv --python 3.10 .venv-reward
git submodule update --init policy_sources/visualnav_transformer
# Install a matching torch/torchvision build supported by the cluster's driver.
# Supply the cluster-approved wheel index/version if required:
"$UV_BIN" pip install --python .venv-reward/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
"$UV_BIN" pip install --python .venv-reward/bin/python -r training/requirements-policy-reward.txt
```

The old `chop` environment exists under
`/fs/nexus-scratch/gershom/anaconda3/envs/chop` (not the active Miniconda root).
It has torch 2.2.0 / transformers 4.40.1 and is left unchanged for SFT jobs.
The reward environment uses CUDA 12.6 wheels; an L40S allocation reported driver
595.71.05 during setup. DINO model/processor files are staged under `HF_HOME`.
Transformers is pinned to 5.15.0 to match the workstation reward checkpoint;
4.56.2 can construct DINOv3 but has incompatible state-dict parameter names.

Prepare/download the checkpoint's DINOv3 model and processor in your Hugging Face
cache on a login/download node if compute nodes have no network. Use your own
authorized Hugging Face access for the DINO model and your existing W&B login or
environment configuration. No tokens are stored in these scripts. W&B project
remains `CHOP`; `WANDB_MODE=offline` works, or set `CHOP_DISABLE_WANDB=1`.
The job checks CUDA, dependencies, policy/reward loading, DINO processor access,
and at least 5 GiB free cache space before warming data.

## I/O and resuming

Before training, the job warms exact resized policy inputs on node-local storage.
The cache directory defaults to `$SLURM_TMPDIR/chop-policy-images-v1`, then
`/tmp/chop-policy-$USER-$SLURM_JOB_ID`. **Do not use the login shell's TMPDIR:**
on Nexus it points to `/gammascratch/gershom/tmp`, an NFS mount. Network-backed
image caches are rejected at startup. Python worker temporary files are also
redirected to `.tmp` under the validated local cache, avoiding NFS socket cleanup
errors. Set `CHOP_SCRATCH_ROOT` or
`CHOP_POLICY_IMAGE_CACHE` if your cluster advertises scratch differently.
Choose actual local SSD storage, not a shared network directory. Budget roughly
2 GB for policy images **plus at least 3 GiB free headroom**. The existing DINO
feature LMDB stays at its configured location; copying its tens of GB to scratch
is optional and is not performed automatically. Cache misses never filter data.

Warm-up happens within each job; it does not use desktop `systemctl` pause/resume
scripts. Ephemeral scratch is rebuilt on another node. Two loading workers
prefetch pinned batches. `CHOP_WORKERS` and `CHOP_CACHE_WORKERS` override the
defaults (2 and 4). Jobs lock their output directory against concurrent writes.

`CHOP_RESUME=auto` resumes `latest.pt` when it exists; otherwise initialization
uses the public checkpoint. Use a new `CHOP_RUN_TAG` for an independent run, or
set `CHOP_OUTPUT` explicitly. `CHOP_RESUME=none` rejects existing run outputs;
an explicit checkpoint path is also accepted. Never point both models at the
same output directory. A scheduler timeout can lose updates after the latest
periodic checkpoint; resubmit with the same paths to continue.

Resume validates the stored configuration, including absolute data/checkpoint
paths. A desktop policy checkpoint with different paths is **not** automatically
portable as a resume checkpoint. Start fresh cluster experiments from the public
weights; do not edit provenance or bypass that guard. Public weights and the
frozen reward checkpoint themselves are portable inputs.

Optional settings: `CHOP_EPOCHS`, `CHOP_BATCH_SIZE`, `CHOP_LR`,
`CHOP_REFERENCE_WEIGHT`, and `CHOP_EVAL_EVERY`. Resume requires the relevant
optimization/data settings to match the saved checkpoint.

Inspect resolved commands without launching training:

```bash
CHOP_SCRATCH_ROOT=/tmp/chop-preview bash sbatch-scripts/finetune-gnm-reward.slurm --dry-run
CHOP_SCRATCH_ROOT=/tmp/chop-preview bash sbatch-scripts/finetune-vint-reward.slurm --dry-run
```

Dry-run prints commands only; it does not validate files, GPU availability or
cluster permissions. Real runs fail early on missing required files.
