#!/usr/bin/env python3
"""Export raw CHOP winner/loser labels for reward-model training.

Unlike the SFT index builder, this script never collapses a scene to one
winning target. Each non-ambiguous human comparison becomes one training row.
The output is JSON and intentionally leaves image loading to the training
dataset, so it can be converted to LMDB/WebDataset later without changing the
preference semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets.preprocess_scand_a_chop import _extract_path

TIE_OR_BAD = {404, 500}


def build_reward_pairs(annotations_dir: Path, images_root: Path, image_ext: str, num_points: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        document = json.loads(annotation_path.read_text())
        bag = Path(document.get("bag", annotation_path.stem)).stem
        for stamp, annotation in (document.get("annotations_by_stamp") or {}).items():
            image_path = images_root / bag / f"img_{stamp}.{image_ext}"
            if not image_path.is_file():
                continue
            paths = annotation.get("paths") or {}
            for comparison in annotation.get("pairwise", []):
                pair = [str(item) for item in comparison.get("pair", [])]
                choice = comparison.get("choice")
                if len(pair) != 2 or str(choice) in {str(value) for value in TIE_OR_BAD}:
                    continue
                winner = str(choice)
                if winner not in pair or any(candidate not in paths for candidate in pair):
                    continue
                loser = pair[1] if winner == pair[0] else pair[0]
                rows.append({
                    "bag": bag,
                    "timestamp": str(stamp),
                    "image_path": str(Path(bag) / image_path.name),
                    "preferred_path": _extract_path(paths[winner], num_points),
                    "rejected_path": _extract_path(paths[loser], num_points),
                    "preferred_id": winner,
                    "rejected_id": loser,
                    "robot_width": annotation.get("robot_width"),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--num-points", type=int, default=8)
    args = parser.parse_args()
    rows = build_reward_pairs(args.annotations_dir, args.images_root, args.image_ext, args.num_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} raw winner/loser comparisons to {args.output}")


if __name__ == "__main__":
    main()
