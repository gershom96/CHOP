from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

_DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "data" / "lora-data" / "train.json"


class OmniVLAChopDataset(Dataset):
    """
    PyTorch Dataset for CHOP counterfactual preferences formatted for OmniVLA.

    Expects an index JSON (defaults to ``data/lora-data/train.json``) where each
    top-level entry contains a ``bag`` name and a ``samples`` list as produced
    by ``preprocess_scand``. The dataset flattens the samples and loads images
    plus trajectory annotations.
    """

    def __init__(
        self,
        image_root: Union[str, Path],
        dataset_path: Union[str, Path, None] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        max_dist: float = 30.0,
        preload_images: bool = False,
        num_points: int = 8
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path is not None else _DEFAULT_INDEX
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Index not found: {self.dataset_path}")

        self._default_transform = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        self.num_points = num_points
        self.image_root = Path(image_root)
        self.max_dist = max_dist

        if transform is None:
            self.transform = self._default_transform
        else:
            self.transform = transform

        self.samples = self._load_index(self.dataset_path)
        self._image_cache: Optional[List[Any]] = [None] * len(self.samples) if preload_images else None

        if preload_images:
            for i, sample in enumerate(self.samples):
                self._image_cache[i] = self._load_image(sample["image_path"])

    def _load_index(self, path: Path) -> List[Dict[str, Any]]:
        raw = json.loads(path.read_text())
        samples: List[Dict[str, Any]] = []

        for bag_entry in raw:
            bag_name = bag_entry.get("bag")
            for sample in bag_entry.get("samples", []):
                item = dict()
                item.setdefault("bag", bag_name)
                item["timestamp"] = sample.get("timestamp")
                item["frame_idx"] = sample.get("frame_idx")
                item["path_0"] = self._convert_path(sample.get("path_0", {}))
                item["path_1"] = self._convert_path(sample.get("path_1", {}))
                if "position" in sample and sample.get("position") is not None:
                    item["position"] = torch.as_tensor(sample.get("position"), dtype=torch.float32)
                if "yaw" in sample and sample["yaw"] is not None:
                    item["yaw"] = float(sample["yaw"])
                item["stop"] = bool(sample.get("stop", False))
                item["image_path"] = sample.get("image_path")

                samples.append(item)

        return samples

    def _resample_path(self, path: torch.Tensor, k: int) -> torch.Tensor:
        if path.numel() == 0:
            return torch.zeros((k, 3), dtype=torch.float32)
        if len(path) == 1:
            return path.repeat(k, 1)

        deltas = path[1:] - path[:-1]
        seg_len = torch.norm(deltas, dim=1)
        cum = torch.cat([torch.zeros(1, device=path.device), torch.cumsum(seg_len, dim=0)])
        total = cum[-1]
        if total == 0:
            return path[0].repeat(k, 1)

        target = torch.linspace(0, total, steps=k, device=path.device)

        out = torch.empty((k, path.size(1)), device=path.device, dtype=path.dtype)
        for i, t in enumerate(target):
            j = torch.searchsorted(cum, t, right=True) - 1
            j = torch.clamp(j, 0, len(seg_len) - 1)
            t0, t1 = cum[j], cum[j + 1]
            alpha = 0.0 if t1 == t0 else float((t - t0) / (t1 - t0))
            out[i] = path[j] * (1 - alpha) + path[j + 1] * alpha
        return out

    def _convert_path(self, path_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:

        def to_tensor(key: str) -> torch.Tensor:
            data = path_data.get(key)
            if not data:
                return torch.empty((0, 3), dtype=torch.float32)
            return torch.as_tensor(data, dtype=torch.float32).reshape(-1, 3)

        return {
            "points": self._resample_path(to_tensor("points"), k=self.num_points),
            # "left_boundary": to_tensor("left_boundary"),
            # "right_boundary": to_tensor("right_boundary"),
        }

    def _load_image(self, rel_path: str) -> Any:
        path = self.image_root / rel_path
        image = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(image)
        arr = torch.from_numpy(np.array(image, dtype=np.uint8))
        return arr.permute(2, 0, 1).float().div(255.0)

    def _get_goal(self, idx: int) -> Dict[str, Any]:
        p0 = self.samples[idx]["path_0"]["points"]
        p1 = self.samples[idx]["path_1"]["points"]

        if p0.numel() == 0 and p1.numel() == 0:
            # fall back to some default goal logic or skip
            path_dist = 0.0
        else:
            p0_dist = float(torch.norm(p0[-1])) if p0.numel() > 0 else 0.0
            p1_dist = float(torch.norm(p1[-1])) if p1.numel() > 0 else 0.0
            path_dist = max(p0_dist, p1_dist)

        if path_dist > self.max_dist:
            path_dist = 0.2 * self.max_dist

        goal_dist = torch.rand(1).item() * (self.max_dist - path_dist) + path_dist
        bag_name = self.samples[idx]["bag"]
        walked = 0.0
        prev_idx = idx
        goal_idx = idx

        # march forward only within this bag
        while True:
            next_idx = goal_idx + 1
            if (
                next_idx >= len(self.samples)
                or self.samples[next_idx]["bag"] != bag_name
            ):
                break  # reached end of this bag

            pos_prev = self.samples[prev_idx]["position"]
            pos_next = self.samples[next_idx]["position"]
            if pos_prev is None or pos_next is None:
                break

            step = float(torch.norm(pos_next - pos_prev))
            walked += step
            prev_idx = next_idx
            goal_idx = next_idx

            if walked >= goal_dist:
                break

        goal_position = self.samples[goal_idx]["position"]
        goal_yaw = self.samples[goal_idx]["yaw"]
        goal_image = (
            self._image_cache[goal_idx]
            if self._image_cache is not None
            else self._load_image(self.samples[goal_idx]["image_path"])
        )

        return {
            "goal_position": goal_position,
            "goal_yaw": goal_yaw,
            "goal_image": goal_image,
        }
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = self._image_cache[idx] if self._image_cache is not None else self._load_image(sample["image_path"])
        goal = self._get_goal(idx)

        # These might need to change based on the model's naming conventions
        return {
            "current_image": image,
            "current_position": sample.get("position"),
            "current_yaw": sample.get("yaw"),
            "goal_image": goal.get("goal_image"),
            "goal_position": goal.get("goal_position"),
            "goal_yaw": goal.get("goal_yaw"),
            "path_0": sample["path_0"],
            "path_1": sample["path_1"],
            "stop": sample.get("stop", False)
        }

def make_dataloader(
    image_root: Union[str, Path],
    dataset_path: Union[str, Path, None] = None,
    transform: Optional[Callable[[Image.Image], Any]] = None,
    num_points: int = 8,
    max_dist: float = 30.0,
    preload_images: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = OmniVLAChopDataset(
        image_root=image_root,
        dataset_path=dataset_path,
        transform=transform,
        max_dist=max_dist,
        preload_images=preload_images,
        num_points=num_points
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
