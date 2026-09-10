"""Observation-only data for reward optimization, without SFT trajectory targets."""

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.policy_image_cache import policy_pixels
from datasets.reward_model_dataset import (
    _FEATURE_ENVS,
    CHOPRewardPreferenceDataset,
    _calibration_for_bag,
)


class PolicyRewardDataset(Dataset):
    def __init__(
        self,
        index,
        split,
        image_root,
        cache,
        calibration,
        limit=0,
        seed=285,
        include_uncached=False,
        policy_image_cache=None,
    ):
        self.image_root = Path(image_root)
        self.policy_image_cache = policy_image_cache
        allowed = {Path(r["bag"]).stem for r in json.loads(Path(split).read_text())}
        self.features = object.__new__(CHOPRewardPreferenceDataset)
        self.features.feature_cache = Path(cache)
        self.features._feature_env = None
        self.features.image_size = (384, 640)
        import lmdb

        cache_key = str(Path(cache).resolve())
        if cache_key not in _FEATURE_ENVS:
            _FEATURE_ENVS[cache_key] = lmdb.open(
                cache_key, readonly=True, lock=False, readahead=False
            )
        self.features._feature_env = _FEATURE_ENVS[cache_key]
        self.calibration = {}
        self.samples = []
        total_observations = 0
        # Read only observation metadata. Preference paths are never targets.
        for bag in json.loads(Path(index).read_text()):
            name = Path(bag["bag"]).stem
            if name not in allowed:
                continue
            frames = {s["frame_idx"]: s["image_path"] for s in bag["samples"]}
            total_observations += len(frames)
            for t in sorted(frames):
                context = [frames.get(t - d) for d in range(5, -1, -1)]
                goal = frames.get(t + 10)
                if all(context) and goal:
                    self.samples.append((name, tuple(context), goal))
            k, transform = _calibration_for_bag(Path(calibration), name)
            k[0] *= 640 / 1280
            k[1] *= 384 / 720
            self.calibration[name] = (k, transform)
        # A fixed random subset is used only when explicitly requested for pilots.
        rng = np.random.default_rng(seed)
        rng.shuffle(self.samples)
        candidates = self.samples
        self.samples = []
        self.cached_paths = set()
        self.include_uncached = include_uncached
        if include_uncached:
            # Cache availability is not an eligibility criterion. Avoid an
            # expensive random-I/O cache scan before training can start.
            self.samples = candidates[:limit] if limit else candidates
        else:
            with self.features._feature_env.begin() as txn:
                cursor = txn.cursor()
                for sample in candidates:
                    if cursor.set_key(sample[1][-1].encode()):
                        self.cached_paths.add(sample[1][-1])
                        self.samples.append(sample)
                        if limit and len(self.samples) >= limit:
                            break
        if not self.samples:
            raise ValueError("No observations with complete context and future goal")
        self.coverage = {
            "indexed_frames": total_observations,
            "complete_context_and_goal": len(candidates),
            "selected_observations": len(self.samples),
            "cache_filter_enabled": not include_uncached,
            "sample_limit": limit,
        }
        self.image = lru_cache(maxsize=512)(self._image)
        self.processor = None

    def _image(self, relative):
        value = policy_pixels(self.image_root, relative, self.policy_image_cache)
        return (
            value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __getstate__(self):
        # Spawn workers: never inherit CUDA state or a parent's LMDB handle.
        state = self.__dict__.copy()
        state.pop("image", None)
        state["features"] = copy.copy(self.features)
        state["features"]._feature_env = None
        state["processor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.image = lru_cache(maxsize=512)(self._image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bag, context, goal = self.samples[index]
        k, transform = self.calibration[bag]
        result = {
            "obs": torch.cat([self.image(p) for p in context]),
            "goal": self.image(goal),
            "intrinsics": k,
            "transform": transform,
            "image_path": context[-1],
        }
        try:
            feature = self.features._cached_feature(context[-1])
        except KeyError:
            if not self.include_uncached:
                raise
            feature = None
        if feature is not None:
            result["feature_map"] = feature
            result["reward_image"] = None
        else:
            from PIL import Image
            from transformers import AutoImageProcessor

            if self.processor is None:
                self.processor = AutoImageProcessor.from_pretrained(
                    "facebook/dinov3-vits16-pretrain-lvd1689m"
                )
            with Image.open(self.image_root / context[-1]) as opened:
                result["reward_image"] = self.processor(
                    opened.convert("RGB"),
                    size={"height": 384, "width": 640},
                    return_tensors="pt",
                )["pixel_values"][0]
            result["feature_map"] = torch.zeros(384, 24, 40, dtype=torch.float16)
        return result


def policy_reward_collate(rows):
    from torch.utils.data import default_collate

    missing = [i for i, row in enumerate(rows) if row["reward_image"] is not None]
    images = [row["reward_image"] for row in rows if row["reward_image"] is not None]
    result = default_collate(
        [{k: v for k, v in row.items() if k != "reward_image"} for row in rows]
    )
    result["missing_features"] = torch.tensor(missing, dtype=torch.long)
    result["reward_image"] = torch.stack(images) if images else None
    return result
