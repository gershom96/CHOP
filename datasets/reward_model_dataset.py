"""Dataset for uncollapsed CHOP pairwise trajectory preferences."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
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
    """Loads image/path winner-loser pairs with calibration-aware resizing.

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
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_root = Path(image_root)
        self.image_size = image_size  # height, width; both must be divisible by 16 for DINOv3
        if image_size[0] % 16 or image_size[1] % 16:
            raise ValueError("image_size must be divisible by 16 for DINOv3")
        allowed_bags = _split_bags(Path(split_index_path) if split_index_path else None)
        rows = json.loads(Path(pairs_path).read_text())
        self.rows = [row for row in rows if allowed_bags is None or Path(row["bag"]).stem in allowed_bags]
        if not self.rows:
            raise ValueError("No reward pairs remain after applying the split")
        self.mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.calibrations = {}
        for bag in {row["bag"] for row in self.rows}:
            self.calibrations[bag] = _calibration_for_bag(Path(calibration_path), bag)

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, path: Path) -> Tuple[Tensor, Tuple[int, int]]:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            original_width, original_height = image.size
            image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std, (original_height, original_width)

    @staticmethod
    def _points(path: Dict[str, Any]) -> Tensor:
        points = torch.as_tensor(path["points"], dtype=torch.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("Reward pair paths must contain Kx3 points")
        return points

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        row = self.rows[index]
        image, (original_height, original_width) = self._image(self.image_root / row["image_path"])
        intrinsics, transform = self.calibrations[row["bag"]]
        # Projection occurs in resized image coordinates, so K must be scaled
        # from the raw SCAND image resolution accordingly.
        scale_x = self.image_size[1] / original_width
        scale_y = self.image_size[0] / original_height
        intrinsics = intrinsics.clone()
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
        return {
            "image": image,
            "preferred_path": self._points(row["preferred_path"]),
            "rejected_path": self._points(row["rejected_path"]),
            "intrinsics": intrinsics,
            "t_cam_from_base": transform,
            "bag": row["bag"],
        }
