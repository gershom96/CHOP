"""Dataset for uncollapsed CHOP pairwise trajectory preferences."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _calibration_for_bag(calibration_path: Path, bag: str) -> Tuple[Tensor, Tensor]:
    """Return K and T_cam_from_base matching the repository visualization code."""
    data = json.loads(calibration_path.read_text())
    robot = "spot" if "Spot" in bag else "jackal" if "Jackal" in bag else None
    if robot is None:
        raise ValueError(f"Cannot infer robot calibration from bag name: {bag}")
    h = data[robot]["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    base_from_cam = torch.eye(4, dtype=torch.float32)
    base_from_cam[:3, :3] = torch.tensor([
        [0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll), math.sin(roll)],
    ])
    base_from_cam[:3, 3] = torch.tensor([h["x"], h["y"], h["z"]], dtype=torch.float32)
    intrinsics = data["Intrinsics"]
    k = torch.tensor([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return k, torch.linalg.inv(base_from_cam)


def _split_bags(index_path: Optional[Path]) -> Optional[set[str]]:
    if index_path is None:
        return None
    return {Path(row["bag"]).stem for row in json.loads(index_path.read_text())}


class CHOPRewardPreferenceDataset(Dataset):
    """Loads one image and all its winner/loser pairs per dataset item.

    The dataset performs no random horizontal flip: flipping an image without
    simultaneously reflecting path coordinates and the camera extrinsics would
    invalidate the physical anchor projection.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        image_root: str | Path,
        calibration_path: str | Path,
        split_index_path: str | Path | None = None,
        image_size: Tuple[int, int] = (384, 640),
        processor_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    ) -> None:
        self.image_root = Path(image_root)
        self.image_size = image_size  # height, width; both must be divisible by 16 for DINOv3
        if image_size[0] % 16 or image_size[1] % 16:
            raise ValueError("image_size must be divisible by 16 for DINOv3")
        allowed_bags = _split_bags(Path(split_index_path) if split_index_path else None)
        rows = json.loads(Path(pairs_path).read_text())
        rows = [row for row in rows if allowed_bags is None or Path(row["bag"]).stem in allowed_bags]
        if not rows:
            raise ValueError("No reward pairs remain after applying the split")
        groups = {}
        for row in rows:
            key = (row["bag"], row["timestamp"], row["image_path"])
            group = groups.setdefault(key, {
                "bag": row["bag"], "image_path": row["image_path"], "comparisons": [],
            })
            group["comparisons"].append((row["preferred_path"], row["rejected_path"]))
        self.groups = list(groups.values())
        try:
            from transformers import AutoImageProcessor
        except ImportError as exc:
            raise ImportError("The reward dataset requires transformers for official DINOv3 preprocessing") from exc
        self.image_processor = AutoImageProcessor.from_pretrained(processor_name)
        self.calibrations = {}
        for bag in {group["bag"] for group in self.groups}:
            self.calibrations[bag] = _calibration_for_bag(Path(calibration_path), bag)

    def __len__(self) -> int:
        return len(self.groups)

    def _image(self, path: Path) -> Tuple[Tensor, Tuple[int, int]]:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            original_width, original_height = image.size
            tensor = self.image_processor(
                image,
                size={"height": self.image_size[0], "width": self.image_size[1]},
                return_tensors="pt",
            )["pixel_values"][0]
        return tensor, (original_height, original_width)

    @staticmethod
    def _points(path: Dict[str, Any]) -> Tensor:
        points = torch.as_tensor(path["points"], dtype=torch.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("Reward pair paths must contain Kx3 points")
        return points

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        group = self.groups[index]
        image, (original_height, original_width) = self._image(self.image_root / group["image_path"])
        intrinsics, transform = self.calibrations[group["bag"]]
        # Projection occurs in resized image coordinates, so K must be scaled
        # from the raw SCAND image resolution accordingly.
        scale_x = self.image_size[1] / original_width
        scale_y = self.image_size[0] / original_height
        intrinsics = intrinsics.clone()
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
        preferred_path = torch.stack([self._points(pair[0]) for pair in group["comparisons"]])
        rejected_path = torch.stack([self._points(pair[1]) for pair in group["comparisons"]])
        return {
            "image": image,
            "preferred_path": preferred_path,
            "rejected_path": rejected_path,
            "intrinsics": intrinsics,
            "t_cam_from_base": transform,
            "bag": group["bag"],
        }


def reward_pair_group_collate(samples):
    """Flatten comparison paths while retaining one image per unique frame."""
    images = torch.stack([sample["image"] for sample in samples])
    intrinsics = torch.stack([sample["intrinsics"] for sample in samples])
    transforms = torch.stack([sample["t_cam_from_base"] for sample in samples])
    pair_image_index = torch.cat([
        torch.full((sample["preferred_path"].shape[0],), index, dtype=torch.long)
        for index, sample in enumerate(samples)
    ])
    return {
        "image": images,
        "intrinsics": intrinsics,
        "t_cam_from_base": transforms,
        "preferred_path": torch.cat([sample["preferred_path"] for sample in samples]),
        "rejected_path": torch.cat([sample["rejected_path"] for sample in samples]),
        "pair_image_index": pair_image_index,
    }
