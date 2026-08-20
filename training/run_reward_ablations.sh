#!/usr/bin/env bash
# Controlled reward-model sweep.  The final SCAND test bags are intentionally
# absent: select a configuration only from the bag-disjoint validation split.
set -euo pipefail

project_root="/home/beast-gamma/Documents/GAMMA/Projects/CHOP"
python_bin="$project_root/.venv-reward/bin/python"
data_root="/media/beast-gamma/Media2"
output_root="$data_root/CHOP/reward_model/ablations_v1"
common=(
  --pairs "$data_root/CHOP/reward_model/raw_pairs.json"
  --image-root "$data_root/Datasets/SCAND/images"
  --calibration "$project_root/evaluation/scand_cameras.json"
  --train-index "$data_root/CHOP/reward_model/splits_v1/train.json"
  --test-index "$data_root/CHOP/reward_model/splits_v1/validation.json"
  --feature-cache "$data_root/CHOP/reward_model/dinov3_feature_cache"
  --batch-size 32 --workers 0 --epochs 10 --patience 2
  --warmup-steps 250 --min-lr-scale 0.1 --weight-decay 0.001
)

run() {
  local name="$1"
  shift
  "$python_bin" -m training.train_trajectory_reward "${common[@]}" \
    --output "$output_root/$name" --wandb-name "chop-rm-$name" "$@"
}

# Isolate score regularisation, learning rate, and head capacity.  All runs
# choose their checkpoint by validation BT loss, never the final test split.
run compact_lr1e4_no_reg --hidden-dim 128 --num-heads 4 --num-layers 1 --lr 1e-4 --score-regularization 0
run compact_lr1e4_reg1e2 --hidden-dim 128 --num-heads 4 --num-layers 1 --lr 1e-4 --score-regularization 0.01
run compact_lr3e5_reg1e2 --hidden-dim 128 --num-heads 4 --num-layers 1 --lr 3e-5 --score-regularization 0.01
run full_lr1e4_reg1e2 --hidden-dim 192 --num-heads 6 --num-layers 2 --lr 1e-4 --score-regularization 0.01
