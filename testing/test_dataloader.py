# scripts/inspect_chop.py
from pathlib import Path
import math
import random
import matplotlib.pyplot as plt
import torch

from datasets.omnivla_chop_dataset import OmniVLAChopDataset


def _tensor_to_image(img: torch.Tensor):
    img = img.detach().cpu()
    if img.ndim == 3 and img.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = torch.clamp(img * std + mean, 0, 1)
        img = img.permute(1, 2, 0)
    return img.numpy()


def plot_bev_and_image(item):
    start_xy = item["current_position"][:2].cpu().numpy()
    goal_rel = item["goal_position"][:2].cpu().numpy() - start_xy
    if item.get("current_yaw") is not None:
        yaw = float(item["current_yaw"])
        c, s = math.cos(-yaw), math.sin(-yaw)  # rotate into robot frame
        x, y = goal_rel
        goal_rel = [c * x - s * y, s * x + c * y]

    fig, (ax_bev, ax_img) = plt.subplots(1, 2, figsize=(10, 4))
    for path, color, label in (
        (item["path_0"]["points"], "tab:green", "path_0"),
        (item["path_1"]["points"], "tab:blue", "path_1"),
    ):
        pts = path.cpu().numpy()
        ax_bev.plot(pts[:, 0], pts[:, 1], marker="o", color=color, label=label)

    ax_bev.scatter([0.0], [0.0], color="red", label="start (robot)")
    ax_bev.scatter([goal_rel[0]], [goal_rel[1]], color="orange", label="goal")
    ax_bev.set_title("BEV paths (robot frame)")
    ax_bev.set_xlabel("x (m)")
    ax_bev.set_ylabel("y (m)")
    ax_bev.grid(True, linestyle="--", alpha=0.3)
    ax_bev.legend()

    img = _tensor_to_image(item["current_image"])
    ax_img.imshow(img)
    ax_img.set_title("Current image")
    ax_img.axis("off")

    fig.tight_layout()
    plt.show()


def main():
    index = Path("data/lora-data/train.json")     # adjust if needed
    image_root = Path("/media/beast-gamma/Media/Datasets/SCAND/images")              # adjust if needed

    ds = OmniVLAChopDataset(image_root=image_root, dataset_path=index, preload_images=False)
    print(f"Loaded Dataset with length: {len(ds)}")
    for i in range(3):
        item = ds[i]
        print(f"\n=== sample {i} ===")
        for k, v in item.items():
            if k.endswith("image"):
                print(f"{k}: tensor {tuple(v.shape)}")
            else:
                print(f"{k}: {v}")
        plot_bev_and_image(item)

if __name__ == "__main__":
    main()
