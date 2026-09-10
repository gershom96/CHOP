"""Shared SSD cache of losslessly resized policy inputs (not DINO features)."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from policy_sources.visualnav_transformer.train.vint_train.data.data_utils import (
    img_path_to_data,
)


def policy_pixels(image_root, relative, cache_root=None):
    # The source dataset is immutable. Version the key if preprocessing changes.
    target = None
    if cache_root:
        key = hashlib.sha256(
            f"policy-pil-v1-85x64:{Path(image_root).resolve()}:{relative}".encode()
        ).hexdigest()
        target = Path(cache_root) / key[:2] / (key + ".npy")
        try:
            pixels = np.load(target, allow_pickle=False)
            if pixels.shape != (3, 64, 85) or pixels.dtype != np.uint8:
                raise ValueError(f"Invalid policy image cache entry: {target}")
            return torch.from_numpy(pixels).float().div_(255)
        except FileNotFoundError:
            pass
    value = img_path_to_data(str(Path(image_root) / relative), (85, 64))
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep headroom on the shared system SSD; a full disk must not corrupt
        # training or cause missing observations to be dropped.
        if shutil.disk_usage(target.parent).free > 3 * 1024**3:
            pixels = value.mul(255).round().to(torch.uint8).numpy()
            fd, name = tempfile.mkstemp(prefix=".building-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    np.save(handle, pixels, allow_pickle=False)
                os.replace(name, target)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
    return value
