# CHOP : Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies

## Repository layout
```
datasets/               # Dataset + dataloader helper stubs
configs/                # YAMLs for data paths + LoRA hparams (default.yaml sample)
policy_sources/         # Upstream policy code copied in wholesale (with licenses)
third_party/NOTICE      # Tracks third-party sources and licenses
data/                   # Your indices/splits (gitignored; create locally)
```

## Install
- initialize git submodules first
- then install in virtualenv

```bash
conda create -n chop python=3.10 -y
conda activate chop
git submodule update --init --recursive
pip install -e .
```

## Using third-party policy code
- Copy upstream visuomotor policy code into `policy_sources/<policy_name>/` with its LICENSE and a short README noting the repo URL and commit hash.
- Keep upstream code unmodified where possible; note any edits in `policy_sources/<policy_name>/CHANGES.md` and update `third_party/NOTICE`.

## Scripts (fill in your logic)
- `scripts/prepare_data.py`: build or mock index JSONs in `data/`.
- `scripts/finetune_policy.py`: load a policy from `policy_sources/`, attach LoRA, and finetune vs. the dataset.
- `scripts/evaluate_policy.py`: load a checkpoint and compute metrics on val/test splits.
- `scripts/visualize_counterfactuals.py`: sanity-check counterfactual annotations.
