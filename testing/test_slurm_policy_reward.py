import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_launcher_rewrite(tmp_path, source):
    """Exercise real Bash handoff without requiring datasets, torch, or CUDA."""
    script = tmp_path / "launcher.sh"
    script.write_text(source)
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\nfrom pathlib import Path\n"
        "root = Path(os.environ['LAUNCH_TEST_ROOT'])\n"
        "if 'training.warm_policy_image_cache' in sys.argv:\n"
        "    (root / 'warming').touch()\n"
        "    deadline = time.monotonic() + 10\n"
        "    while not (root / 'continue').exists():\n"
        "        if time.monotonic() > deadline: sys.exit(9)\n"
        "        time.sleep(0.01)\n"
        "if 'training.finetune_policy_reward' in sys.argv:\n"
        "    (root / 'training-started').touch()\n"
    )
    stub.chmod(0o700)
    required = tmp_path / "input"
    required.touch()
    cache = tmp_path / "features"
    cache.mkdir()
    (cache / "data.mdb").touch()
    env = environment(tmp_path)
    env.update(LAUNCH_TEST_ROOT=str(tmp_path), CHOP_PYTHON=str(stub),
               CHOP_FEATURE_CACHE=str(cache), CHOP_IMAGE_ROOT=str(tmp_path),
               CHOP_OUTPUT=str(tmp_path / "output"))
    for key in ("INDEX", "TRAIN_SPLIT", "VAL_SPLIT", "PUBLIC_CHECKPOINT",
                "REWARD_CHECKPOINT", "CALIBRATION"):
        env[f"CHOP_{key}"] = str(required)
    process = subprocess.Popen(["bash", str(script), "gnm"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "warming").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("Launcher did not reach warm-up")
            time.sleep(0.01)
        # Exact shortening made by the checkout update during the failed jobs.
        script.write_text(source.replace("/gammascratch/gershom/hf_cache", "/tmp/hf"))
        (tmp_path / "continue").touch()
        stdout, stderr = process.communicate(timeout=10)
        return process.returncode, (tmp_path / "training-started").exists(), stdout, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_trainer_handoff_survives_launcher_update_during_warmup(tmp_path):
    result = run_launcher_rewrite(
        tmp_path, (ROOT / "sbatch-scripts/run_policy_reward.sh").read_text()
    )
    assert result[0] == 0, result
    assert result[1], result
    assert "starting reward-policy optimization" in result[2]


def environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHOP_")}
    env.update(
        CHOP_PROJECT_ROOT=str(ROOT),
        CHOP_DATA_ROOT=str(tmp_path / "cluster data"),
        CHOP_SCRATCH_ROOT=str(tmp_path / "local scratch"),
    )
    return env


@pytest.mark.parametrize("model", ["gnm", "vint"])
def test_sbatch_launcher_is_regular_file(model):
    launcher = ROOT / f"sbatch-scripts/finetune-{model}-reward.slurm"
    assert launcher.is_file()
    assert not launcher.is_symlink()
    assert (ROOT / "sbatch-scripts/run_policy_reward.sh").is_file()
    assert not (ROOT / "slurm").exists()


@pytest.mark.parametrize("model", ["gnm", "vint"])
def test_cluster_commands_use_cluster_defaults_and_full_data(tmp_path, model):
    env = environment(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / f"sbatch-scripts/finetune-{model}-reward.slurm"), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--model") + 1] == model
    assert (
        command[command.index("--image-root") + 1] == "/fs/gamma-datasets/SCAND/images"
    )
    assert command[command.index("--index") + 1] == str(
        tmp_path / "cluster data/lora-data/train.json"
    )
    assert command[command.index("--policy-image-cache") + 1] == str(
        tmp_path / "local scratch/chop-policy-images-v1"
    )
    assert command[command.index("--train-limit") + 1] == "0"
    assert command[command.index("--reward-checkpoint") + 1] == str(
        ROOT / "weights/trajectory_reward/compact_lr1e4_no_reg/best.pt"
    )
    assert command[command.index("--val-limit") + 1] == "0"
    assert "--include-uncached" in command
    assert "/media/beast-gamma" not in result.stdout
    assert not (tmp_path / "local scratch").exists()


def test_resume_and_model_checkpoint_overrides(tmp_path):
    env = environment(tmp_path)
    output = tmp_path / "existing output"
    output.mkdir()
    (output / "latest.pt").touch()
    env.update(
        CHOP_OUTPUT=str(output),
        CHOP_PUBLIC_CHECKPOINT="/cluster/public/vint.pth",
        CHOP_DISABLE_WANDB="1",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "sbatch-scripts/run_policy_reward.sh"), "vint", "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--resume") + 1] == str(output / "latest.pt")
    assert command[command.index("--checkpoint") + 1] == "/cluster/public/vint.pth"
    assert "--disable-wandb" in command


def test_missing_scratch_is_not_silently_replaced_with_network_cache(tmp_path):
    env = environment(tmp_path)
    for key in ("CHOP_SCRATCH_ROOT", "SLURM_TMPDIR", "TMPDIR", "SLURM_JOB_ID"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(ROOT / "sbatch-scripts/run_policy_reward.sh"), "gnm", "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "node-local" in result.stderr


def test_nexus_job_does_not_use_nfs_tmpdir(tmp_path):
    env = environment(tmp_path)
    env.pop("CHOP_SCRATCH_ROOT")
    env.pop("SLURM_TMPDIR", None)
    env.update(SLURM_JOB_ID="12345", TMPDIR="/gammascratch/gershom/tmp", USER="gershom")
    result = subprocess.run(
        ["bash", str(ROOT / "sbatch-scripts/finetune-gnm-reward.slurm"), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    command = shlex.split(result.stdout.splitlines()[-1])
    local_root = next(
        (p for p in ("/scratch1", "/scratch0") if Path(p).is_dir() and os.access(p, os.W_OK)),
        "/tmp",
    )
    assert command[command.index("--policy-image-cache") + 1] == (
        local_root + "/chop-policy-gershom-12345"
    )
    assert (
        command[command.index("--feature-cache") + 1]
        == "/gammascratch/gershom/CHOP/reward_model/dinov3_feature_cache"
    )
    script = (ROOT / "sbatch-scripts/run_policy_reward.sh").read_text()
    assert 'export TMPDIR="$image_cache/.tmp"' in script
    assert script.index('cache_fs=') < script.index('export TMPDIR=')


@pytest.mark.parametrize("model,gpu", [("gnm", "l40s"), ("vint", "rtxa5000")])
def test_gpu_type_matches_original_nexus_scripts(model, gpu):
    script = (ROOT / f"sbatch-scripts/finetune-{model}-reward.slurm").read_text()
    assert f"#SBATCH --gres=gpu:{gpu}:1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --time=3-00:00:00" in script
    assert "#SBATCH --signal=B:USR1@300" in script
