#!/usr/bin/env bash
set -euo pipefail
cd /home/beast-gamma/Documents/GAMMA/Projects/CHOP
exec .venv-reward/bin/python -u -m training.finetune_policy_reward \
  --model gnm --checkpoint /media/beast-gamma/Media2/Projects/CHOP/weights/gnm.pth \
  --reward-checkpoint /media/beast-gamma/Media2/CHOP/reward_model/ablations_v1/compact_lr1e4_no_reg/best.pt \
  --index /media/beast-gamma/Media2/CHOP/lora-data/train.json \
  --train-split /media/beast-gamma/Media2/CHOP/reward_model/splits_v1/train.json \
  --val-split /media/beast-gamma/Media2/CHOP/reward_model/splits_v1/validation.json \
  --image-root /media/beast-gamma/Media2/Datasets/SCAND/images \
  --feature-cache /media/beast-gamma/Media2/CHOP/reward_model/dinov3_feature_cache \
  --calibration evaluation/scand_cameras.json \
  --output /media/beast-gamma/Media2/CHOP/reward_model/policy_reward/gnm_pilot_v1 \
  --steps 300 --train-limit 512 --val-limit 64 --batch-size 8 --eval-every 50
