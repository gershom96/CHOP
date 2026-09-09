"""Observation-only data for reward optimization, without SFT trajectory targets."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.reward_model_dataset import (
    _FEATURE_ENVS,
    CHOPRewardPreferenceDataset,
    _calibration_for_bag,
)
from policy_sources.visualnav_transformer.train.vint_train.data.data_utils import (
    img_path_to_data,
)


class PolicyRewardDataset(Dataset):
    def __init__(self, index, split, image_root, cache, calibration, limit=0, seed=285):
        self.image_root = Path(image_root)
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
        # Read only observation metadata. Preference paths are never targets.
        for bag in json.loads(Path(index).read_text()):
            name = Path(bag["bag"]).stem
            if name not in allowed:
                continue
            frames = {s["frame_idx"]: s["image_path"] for s in bag["samples"]}
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
        with self.features._feature_env.begin() as txn:
            cursor = txn.cursor()
            for sample in candidates:
                if cursor.set_key(sample[1][-1].encode()):
                    self.samples.append(sample)
                    if limit and len(self.samples) >= limit:
                        break
        if not self.samples:
            raise ValueError("No observations with complete context and future goal")
        self.image = lru_cache(maxsize=512)(self._image)

    def _image(self, relative):
        value = img_path_to_data(str(self.image_root / relative), (85, 64))
        return (
            value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bag, context, goal = self.samples[index]
        k, transform = self.calibration[bag]
        return {
            "obs": torch.cat([self.image(p) for p in context]),
            "goal": self.image(goal),
            "feature_map": self.features._cached_feature(context[-1]),
            "intrinsics": k,
            "transform": transform,
            "image_path": context[-1],
        }
