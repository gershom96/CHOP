"""Populate exact policy-input SSD cache in bag/time order, with bounded RAM."""

import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from datasets.policy_image_cache import policy_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = sorted(
        {
            row["image_path"]
            for bag in json.loads(Path(args.index).read_text())
            for row in bag["samples"]
        }
    )
    if args.limit:
        paths = paths[: args.limit]
    print(json.dumps({"cache_images": len(paths), "workers": args.workers}), flush=True)
    started = time.monotonic()

    def warm(path):
        policy_pixels(args.image_root, path, args.cache)

    # Keep I/O near adjacent source files instead of queuing random bags.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = deque()
        source = iter(paths)
        for _ in range(args.workers * 2):
            path = next(source, None)
            if path is not None:
                pending.append(pool.submit(warm, path))
        completed = 0
        while pending:
            pending.popleft().result()
            completed += 1
            path = next(source, None)
            if path is not None:
                pending.append(pool.submit(warm, path))
            if completed % 256 == 0 or completed == len(paths):
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "cached": completed,
                            "total": len(paths),
                            "seconds": elapsed,
                            "images_per_second": completed / elapsed,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
