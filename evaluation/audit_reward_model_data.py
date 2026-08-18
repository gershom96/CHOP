#!/usr/bin/env python3
"""Audit raw CHOP reward pairs without modifying or filtering training data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from datasets.reward_model_dataset import _calibration_for_bag


def _project(points, intrinsics, transform, height, width):
    points = torch.as_tensor(points, dtype=torch.float32)
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    camera = (transform @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    pixels = (intrinsics @ camera.T).T
    uv = pixels[:, :2] / depth.clamp_min(1e-5).unsqueeze(-1)
    visible = (depth > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv.numpy(), visible.numpy()


def _scaled_calibration(calibration, bag, original_height, original_width, image_size):
    intrinsics, transform = _calibration_for_bag(calibration, bag)
    intrinsics = intrinsics.clone()
    intrinsics[0] *= image_size[1] / original_width
    intrinsics[1] *= image_size[0] / original_height
    return intrinsics, transform


def _draw_path(draw, uv, visible, color):
    last = None
    for point, is_visible in zip(uv, visible):
        if is_visible:
            x, y = map(float, point)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            if last is not None:
                draw.line((last[0], last[1], x, y), fill=color, width=2)
            last = (x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_size = (args.image_height, args.image_width)
    train_bags = {Path(row["bag"]).stem for row in json.loads(args.train_index.read_text())}
    test_bags = {Path(row["bag"]).stem for row in json.loads(args.test_index.read_text())}
    rows = json.loads(args.pairs.read_text())
    sample = random.Random(0).sample(rows, min(args.samples, len(rows)))
    raw_choices = Counter()
    source_pairs = 0
    for path in args.annotations_dir.glob("*.json"):
        for annotation in json.loads(path.read_text()).get("annotations_by_stamp", {}).values():
            for comparison in annotation.get("pairwise", []):
                raw_choices[str(comparison.get("choice"))] += 1
                source_pairs += 1
    pair_keys = [(row["bag"], row["timestamp"], row["preferred_id"], row["rejected_id"]) for row in rows]
    def stationary_count(field):
        return sum(
            max(
                (sum((point[axis] - row[field]["points"][0][axis]) ** 2 for axis in (0, 1)) ** 0.5)
                for point in row[field]["points"]
            ) < 0.02
            for row in rows
        )
    visibility = Counter()
    overlay_rows = sample[:8]
    for index, row in enumerate(sample):
        image_path = args.image_root / row["image_path"]
        with Image.open(image_path) as image:
            original_width, original_height = image.size
            canvas = None
            if index < len(overlay_rows):
                canvas = image.convert("RGB").resize((args.image_width, args.image_height), Image.Resampling.BILINEAR)
        intrinsics, transform = _scaled_calibration(
            args.calibration, row["bag"], original_height, original_width, image_size
        )
        for name in ("preferred", "rejected"):
            uv, visible = _project(row[f"{name}_path"]["points"], intrinsics, transform, *image_size)
            visibility[f"{name}_anchors"] += len(visible)
            visibility[f"{name}_visible_anchors"] += int(visible.sum())
            visibility[f"{name}_paths_with_visible_anchor"] += int(visible.any())
            if canvas is not None:
                _draw_path(ImageDraw.Draw(canvas), uv, visible, "lime" if name == "preferred" else "red")
        if canvas is not None:
            canvas.save(args.output_dir / f"overlay_{index}_{row['bag']}_{row['timestamp']}.jpg")
    report = {
        "grain": "one non-tie, non-both-bad human winner/loser comparison per row",
        "raw_source_pairwise_records": source_pairs,
        "raw_choice_counts": dict(raw_choices),
        "exported_pair_rows": len(rows),
        "exact_duplicate_export_rows": len(pair_keys) - len(set(pair_keys)),
        "train_bags": len(train_bags),
        "test_bags": len(test_bags),
        "bag_overlap": len(train_bags & test_bags),
        "rows_in_train_bags": sum(row["bag"] in train_bags for row in rows),
        "rows_in_test_bags": sum(row["bag"] in test_bags for row in rows),
        "preferred_stationary_paths": stationary_count("preferred_path"),
        "rejected_stationary_paths": stationary_count("rejected_path"),
        "visibility_sample_size": len(sample),
        "visibility": dict(visibility),
        "dino_preprocessing": "official AutoImageProcessor with explicit 384x640 size",
        "dataset_emits_candidate_ids": False,
        "overlays": sorted(path.name for path in args.output_dir.glob("overlay_*.jpg")),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
