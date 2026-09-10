# Nexus checkout and launcher audit — September 10, 2026

Inspected over the user's authenticated SSH connection to `nexusgamma`:
`/fs/nexus-scratch/gershom/CHOP`.

## Changes and conflicts

- Cluster initially on clean `main`, commit
  `d46b2b9e07586ac512e10972dc4d5a4d68cbe3f7`.
- Local reward branch HEAD: `fe39edcdd7176fa03ab8eb9276bd3f08c6533bfc`,
  plus the uncommitted full-data/I/O/Slurm work.
- Common ancestor: `deda7478fd71f201d40adcbb8e45c0b49b0f8254`.
- Cluster-only committed file changes: `configs/chop_nomad_vnt.yaml` sets
  `clipping: True` and both maximum distance categories to 10 instead of 20.
  A non-mutating `git merge-tree` check reported no merge conflicts. Those
  cluster settings are preserved; no wholesale branch merge is performed.
- The visualnav submodule commits differ, but their GNM/ViNT model code,
  base model, and image preprocessing code are identical. Differences are in
  deployment/ROS2 files; the cluster submodule is not changed.
- The existing `sbatch-scripts/` directory was initially ignored by `.gitignore` line 6
  in both checkouts. Its four SFT scripts were not tracked or locally available.
  They are preserved, with new reward aliases alongside them. Canonical reward
  scripts live under the non-ignored `slurm/` directory.
  Both checkouts now expose the same two reward symlinks, with narrow Git ignore
  exceptions for those aliases only; the older SFT scripts remain ignored.

## Subsequent local layout cleanup

The local reward launchers, shared script, and docs were consolidated directly
into `sbatch-scripts/`, replacing the aliases and removing the local `slurm/`
directory. Git ignore exceptions cover these five reward files only. The cleanup
was subsequently pulled on Nexus. Earlier uncommitted deployment files are
preserved in stash `before-reward-branch-sync-20260910`; the cluster's NoMaD
configuration commit was merged without conflicts. Its submodule is unchanged.

## Dataset reconciliation and environment setup

- Local and cluster `data/lora-data/train.json` SHA256:
  `fab73a9faee0f4f4cd18f7bb863d1f959fc6c4bc5eb80029cb2f6c9351621a5a`.
- Local and cluster `test.json` SHA256:
  `35968fc15c5610e3e5d92326f7a43e0f31ce13f9afd1589f14968386a14f4781`.
- Development index: 99 bags, 101,514 distinct image paths; every referenced
  image file exists on Nexus. Of these, 98,474 observations have the required
  six-frame context and future goal. Reward splits partition the 99 bags into
  79 training and 20 validation bags with no overlap or missing bags.
- Held-out test index: 24 bags, 28,602 distinct image paths, no overlap with
  development bags. All split JSON hashes also match across machines.
- Both image roots contain the same 124 bag directory names. The sole directory
  outside the CHOP indices is `A_Spot_Union_Union_Wed_Nov_10_67`, present on both
  machines. Additional raw files are not automatically additional labeled data;
  the original split is retained. This audit verifies indexed training-file
  existence and index identity, not byte equality of all raw images.
- Reproduce bag/frame/split coverage with `testing/audit_policy_dataset.py`.
- Existing `chop` is under `/fs/nexus-scratch/gershom/anaconda3/envs/chop`,
  with torch 2.2.0 and transformers 4.40.1. It is left unchanged.
- New `.venv-reward`: UV-managed Python 3.10, torch 2.8.0+cu126,
  torchvision 0.23.0, transformers 5.15.0. Transformers 4.56.2 failed strict
  checkpoint loading because its DINO parameter names differ; 5.15.0 matches
  the workstation. Dependencies are in `training/requirements-policy-reward.txt`.
- Reward checkpoint transferred to
  `weights/trajectory_reward/compact_lr1e4_no_reg/best.pt`; SHA256 matches:
  `aaab3a24df5e21b671a049122625aa3061f81fbf9f3f192e404e215e7a0087ce`.
- Completed LMDB transferred to the documented gamma-scratch path using rsync;
  it opens read-only with 58,452 entries. DINO model/processor files are staged
  under `/gammascratch/gershom/CHOP/huggingface` and then copied into the existing
  shell-configured `/gammascratch/gershom/hf_cache`; no credentials were copied.
- The workstation source disk was unmounted at setup time and was mounted
  read-only at `/media/beast-gamma/Media2` for transfer.

## Verified cluster conventions

- Existing GNM SFT job: `gpu:l40s:4`, 96G memory, 43 hours, 8 tasks.
- Existing ViNT SFT job: `gpu:rtxa5000:8`, 128G memory, 47 hours, 8 tasks.
- Both: `gamma` account/partition, `huge-long` QoS, old Conda `chop` environment.
- New reward jobs keep GPU types/account/partition/QoS/time limits but use one
  GPU, one task, six CPUs and 48G: the reward policy trainer is not DDP.
  They use a separate UV environment, as requested, not the old Conda environment.
- Public weights and original train/test indices exist under this checkout.
  SCAND image root is `/fs/gamma-datasets/SCAND/images`.
- `TMPDIR=/gammascratch/gershom/tmp` is NFS. New scripts use advertised Slurm
  scratch, otherwise writable `/scratch1`, `/scratch0`, then `/tmp`. Worker
  temporary files are redirected under that validated local cache as well.
- At inspection `/fs/nexus-scratch` had about 33 GiB free, and the user's gamma
  scratch had about 74 GiB. The default DINO LMDB destination is therefore
  `/gammascratch/gershom/CHOP/reward_model/dinov3_feature_cache`.

## Narrow deployment scope

Only reward-policy runtime modules, canonical launchers, related tests/docs and
small calibration/split metadata are deployed. The only existing tracked runtime
file changed is `datasets/__init__.py`: the same exported dataset names are made
lazy so standalone reward training does not import the OmniVLA stack eagerly.
The four old SFT sbatch scripts, NoMaD config, submodules, existing weights/data,
and unrelated local visualization/evaluation changes are not overwritten.

The initial audit used only scheduler `--test-only` checks. At the user's later
request, checkpoint/cache transfer and UV setup were completed and real jobs
were submitted. GNM and ViNT each passed a two-update real-data GPU smoke test:
finite objectives, nonzero gradients, and slight reward gains on eight sampled
validation observations. This validates execution, not full-data improvement.

Initial full-job preflight failures exposed a nearly full `/tmp` and an inherited
`HF_HOME` different from the staging location. Both were fixed before optimization.
Full-data submissions: GNM `7490936`, ViNT `7490937`, using one L40S GPU each.
Each requests three epochs over 79,361 training and 19,113 validation observations;
the ViNT submission overrides its RTX A5000 default with `--gres=gpu:l40s:1`.
