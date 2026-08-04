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

## ICRA revision experiments

The dataset preprocessor supports controlled targets needed to distinguish
counterfactual human preferences from extra fine-tuning data.  Keep the train,
test, model, and optimization settings identical across runs.

```bash
# Repeat once per mode: preferred, original, human_guided, random, all,
# geometric_progress. Each invocation writes train.json and test.json.
python datasets/preprocess_scand_a_chop.py \
  --scand-dir data/annotations/preferences --images-root data/images \
  --output-dir data/ablations/original --target-mode original

# Report label ambiguity and agreement. The default reads annotator_id from
# each export, with a parent-directory fallback.
python evaluation/analyze_annotation_quality.py data/annotations/multi_annotator \
  --output outputs/annotation_quality.json
```

For OmniVLA, set `preference_ranking_weight > 0` in the training configuration
to add the direct Bradley--Terry pairwise trajectory-ranking objective. Set it
to zero for the original best-trajectory SFT result. Report both objectives
and all six preprocessing modes rather than attributing improvements solely to
the preferred target.

#
