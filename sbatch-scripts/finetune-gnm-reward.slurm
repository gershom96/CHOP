#!/usr/bin/env bash
#SBATCH --job-name=chop-gnm-reward
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-19:00:00

set -euo pipefail
project_root="${CHOP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$project_root/sbatch-scripts/run_policy_reward.sh" gnm "$@"
