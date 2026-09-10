#!/usr/bin/env bash
# Shared cluster launcher for reward-only GNM / ViNT optimization, not BT training.
set -euo pipefail
umask 077
model="${1:?Usage: run_policy_reward.sh gnm|vint [--dry-run]}"
shift
case "$model" in gnm|vint) ;; *) echo "Unsupported model: $model" >&2; exit 2 ;; esac
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if (( $# )); then echo "Unexpected arguments: $*" >&2; exit 2; fi

project_root="${CHOP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
project_root="$PWD"
data_root="${CHOP_DATA_ROOT:-$project_root/data}"
reward_root="${CHOP_REWARD_ROOT:-$data_root/reward_model}"
python="${CHOP_PYTHON:-$project_root/.venv-reward/bin/python}"
index="${CHOP_INDEX:-$data_root/lora-data/train.json}"
train_split="${CHOP_TRAIN_SPLIT:-$reward_root/splits_v1/train.json}"
val_split="${CHOP_VAL_SPLIT:-$reward_root/splits_v1/validation.json}"
image_root="${CHOP_IMAGE_ROOT:-/fs/gamma-datasets/SCAND/images}"
checkpoint="${CHOP_PUBLIC_CHECKPOINT:-$project_root/weights/$model.pth}"
reward_checkpoint="${CHOP_REWARD_CHECKPOINT:-$project_root/weights/trajectory_reward/compact_lr1e4_no_reg/best.pt}"
# Model files only, staged from the authorized workstation cache; no credentials.
export HF_HOME="${HF_HOME:-/gammascratch/gershom/hf_cache}"
# The Nexus project allocation has less free space than the ~41 GiB LMDB.
feature_cache="${CHOP_FEATURE_CACHE:-/gammascratch/gershom/CHOP/reward_model/dinov3_feature_cache}"
calibration="${CHOP_CALIBRATION:-$project_root/evaluation/scand_cameras.json}"
output="${CHOP_OUTPUT:-$reward_root/policy_reward/${model}_${CHOP_RUN_TAG:-cluster_reward_v1}}"
scratch_root="${CHOP_SCRATCH_ROOT:-${SLURM_TMPDIR:-}}"
if [[ -n "${CHOP_POLICY_IMAGE_CACHE:-}" ]]; then
  image_cache="$CHOP_POLICY_IMAGE_CACHE"
elif [[ -n "$scratch_root" ]]; then
  image_cache="$scratch_root/chop-policy-images-v1"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Nexus TMPDIR is NFS and /tmp may be full; prefer the dedicated local disks.
  local_root=/tmp
  for candidate in /scratch1 /scratch0; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      local_root="$candidate"
      break
    fi
  done
  image_cache="$local_root/chop-policy-${USER:-$(id -un)}-${SLURM_JOB_ID}"
else
  echo 'Set CHOP_SCRATCH_ROOT to node-local scratch (or CHOP_POLICY_IMAGE_CACHE). No local scratch was advertised.' >&2
  exit 2
fi

resume="${CHOP_RESUME:-auto}"
resume_args=()
if [[ "$resume" == auto ]]; then
  if [[ -f "$output/latest.pt" ]]; then resume_args=(--resume "$output/latest.pt"); fi
elif [[ "$resume" != none ]]; then
  resume_args=(--resume "$resume")
fi

warm=("$python" -u -m training.warm_policy_image_cache
  --index "$index" --image-root "$image_root" --cache "$image_cache"
  --workers "${CHOP_CACHE_WORKERS:-4}")
train=("$python" -u -m training.finetune_policy_reward
  --model "$model" --checkpoint "$checkpoint" --reward-checkpoint "$reward_checkpoint"
  --index "$index" --train-split "$train_split" --val-split "$val_split"
  --image-root "$image_root" --feature-cache "$feature_cache" --calibration "$calibration"
  --output "$output" --epochs "${CHOP_EPOCHS:-3}" --train-limit 0 --val-limit 0
  --include-uncached --batch-size "${CHOP_BATCH_SIZE:-8}" --eval-every "${CHOP_EVAL_EVERY:-200}"
  --monitor-limit 64 --lr "${CHOP_LR:-1e-6}" --reference-weight "${CHOP_REFERENCE_WEIGHT:-10}"
  --workers "${CHOP_WORKERS:-2}" --policy-image-cache "$image_cache" "${resume_args[@]}")
if [[ "${CHOP_DISABLE_WANDB:-0}" == 1 ]]; then train+=(--disable-wandb); fi

printf 'Project: %s\nOutput: %s\nImage cache: %s\n' "$project_root" "$output" "$image_cache"
if (( dry_run )); then
  echo 'Dry run: commands only; files, environment and GPU are not validated.'
  printf '%q ' "${warm[@]}"; printf '\n'
  printf '%q ' "${train[@]}"; printf '\n'
  exit 0
fi

[[ -x "$python" ]] || { echo "Missing UV-environment Python: $python" >&2; exit 2; }
for file in "$index" "$train_split" "$val_split" "$checkpoint" "$reward_checkpoint" "$calibration" "$feature_cache/data.mdb"; do
  [[ -r "$file" ]] || { echo "Missing required input: $file" >&2; exit 2; }
done
[[ -d "$image_root" ]] || { echo "Missing image root: $image_root" >&2; exit 2; }
if (( ${#resume_args[@]} )); then
  [[ -r "${resume_args[1]}" ]] || { echo "Missing resume checkpoint: ${resume_args[1]}" >&2; exit 2; }
elif [[ -e "$output/latest.pt" || -e "$output/metrics.jsonl" ]]; then
  echo 'Existing run output detected without resume. Choose a new CHOP_RUN_TAG or CHOP_OUTPUT.' >&2
  exit 2
fi
mkdir -p "$output" "$image_cache"
[[ -O "$image_cache" ]] || { echo "Image cache must be owned by the current user: $image_cache" >&2; exit 2; }
cache_fs="$(stat -f -c %T "$image_cache")"
case "$cache_fs" in
  nfs*|cifs|smb*|lustre|gpfs|afs)
    echo "Refusing network-backed image cache ($cache_fs): $image_cache. Set CHOP_SCRATCH_ROOT to node-local storage." >&2
    exit 2
    ;;
esac
# Python worker sockets and cleanup must not inherit Nexus's NFS TMPDIR either.
mkdir -p "$image_cache/.tmp"
export TMPDIR="$image_cache/.tmp"
# Prevent two submissions from writing the same checkpoint and metrics files.
exec 9>"$output/.training.lock"
flock -n 9 || { echo "Another job owns output: $output" >&2; exit 2; }
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Slurm owns CUDA_VISIBLE_DEVICES. Do not override its GPU allocation.
"$python" - "$checkpoint" "$reward_checkpoint" "$image_cache" "$model" <<'PY'
import shutil
import sys
import torch
import transformers
import lmdb
import efficientnet_pytorch
import warmup_scheduler
import wandb
from training.finetune_policy_reward import load_policy
from training.policy_reward import FrozenPolicyReward

assert torch.cuda.is_available(), "CUDA unavailable in reward environment"
assert tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 56), "DINOv3 requires transformers >=4.56"
assert shutil.disk_usage(sys.argv[3]).free >= 5 * 1024**3, "Provide at least 5 GiB free on policy-cache storage"
policy = load_policy(sys.argv[1])
assert sys.argv[4] in type(policy).__name__.lower(), "Public checkpoint architecture mismatch"
reward = FrozenPolicyReward(sys.argv[2])
# Verify model/processor access now, not after an expensive cache warm-up.
transformers.AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
print("Preflight passed:", "torch", torch.__version__, "transformers", transformers.__version__,
      "GPU", torch.cuda.get_device_name(0), flush=True)
PY
"${warm[@]}"
exec "${train[@]}"
