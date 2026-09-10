"""Read-only bag/frame coverage audit for the reward-policy dataset."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--image-root", type=Path, required=True)
    args = parser.parse_args()
    bag_sets = {}
    results = {}
    for split in ("train", "test"):
        path = args.project / "data/lora-data" / f"{split}.json"
        rows = json.loads(path.read_text())
        names = {Path(row["bag"]).stem for row in rows}
        bag_sets[split] = names
        paths = {s["image_path"] for row in rows for s in row["samples"]}
        eligible = 0
        for row in rows:
            frames = {s["frame_idx"] for s in row["samples"]}
            eligible += sum(
                all(t - d in frames for d in range(6)) and t + 10 in frames
                for t in frames
            )
        results[split] = {
            "bag_records": len(rows), "unique_bags": len(names),
            "sample_records": sum(len(row["samples"]) for row in rows),
            "unique_image_paths": len(paths),
            "complete_context_and_goal": eligible,
        }
    directories = {p.name for p in args.image_root.iterdir() if p.is_dir()}
    indexed = bag_sets["train"] | bag_sets["test"]
    results["image_root"] = {
        "directories": len(directories),
        "unindexed_directories": sorted(directories - indexed),
        "indexed_bags_without_directory": sorted(indexed - directories),
    }
    results["train_test_bag_overlap"] = sorted(bag_sets["train"] & bag_sets["test"])
    reward_splits = {}
    for name in ("train", "validation", "test"):
        rows = json.loads((args.project / "data/reward_model/splits_v1" / f"{name}.json").read_text())
        reward_splits[name] = {Path(row["bag"]).stem for row in rows}
    results["reward_split_bags"] = {k: len(v) for k, v in reward_splits.items()}
    results["reward_train_val_overlap"] = sorted(reward_splits["train"] & reward_splits["validation"])
    results["reward_train_val_vs_policy_index_difference"] = sorted(
        (reward_splits["train"] | reward_splits["validation"]) ^ bag_sets["train"]
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
