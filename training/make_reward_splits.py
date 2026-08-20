#!/usr/bin/env python3
"""Create fixed bag-disjoint train/validation/test splits for reward learning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True,
                        help="Existing bag-disjoint training index.")
    parser.add_argument("--test-index", type=Path, required=True,
                        help="Existing held-out test index; it remains untouched.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=285)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be strictly between zero and one")

    candidates = json.loads(args.train_index.read_text())
    final_test = json.loads(args.test_index.read_text())

    def compact_bags(rows: list[dict]) -> list[dict[str, str]]:
        return [{"bag": bag} for bag in sorted({Path(row["bag"]).stem for row in rows})]

    # The original lora-data indexes repeat the bag on every annotation row
    # (hundreds of MB).  Reward datasets need only the bag set, so write a
    # compact index that avoids that recurring startup and memory cost.
    candidates = compact_bags(candidates)
    final_test = compact_bags(final_test)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    val_count = max(1, round(len(candidates) * args.validation_fraction))
    validation, train = candidates[:val_count], candidates[val_count:]

    def bags(rows: list[dict]) -> set[str]:
        return {Path(row["bag"]).stem for row in rows}

    train_bags, val_bags, test_bags = bags(train), bags(validation), bags(final_test)
    if train_bags & val_bags or train_bags & test_bags or val_bags & test_bags:
        raise RuntimeError("splits are not bag-disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", final_test)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "validation_fraction_of_original_train": args.validation_fraction,
        "bags": {"train": len(train_bags), "validation": len(val_bags), "test": len(test_bags)},
    }, indent=2) + "\n")
    print(json.dumps({"train_bags": len(train_bags), "validation_bags": len(val_bags), "test_bags": len(test_bags)}))


if __name__ == "__main__":
    main()
