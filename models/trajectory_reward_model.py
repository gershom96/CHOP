"""Trajectory-conditioned visual reward model for CHOP.

Each local path waypoint is an anchor query: its metric base-frame position is
projected through the calibrated camera model, then it samples visual features
at learned offsets around that image reference.  This is a lightweight,
Deformable-DETR-style alternative to rasterising a trajectory into a mask.

The model scores *one* trajectory at a time::

    score = reward_model(image, path_points, intrinsics, T_cam_from_base)

It can therefore be trained on raw winner/loser labels using
``bradley_terry_loss`` and used to rerank candidates from any policy.  It is
not a policy and should not be used to produce actions directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def bradley_terry_loss(
    preferred_scores: Tensor,
    rejected_scores: Tensor,
    temperature: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Negative Bradley--Terry log-likelihood and pairwise accuracy."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Keep the likelihood in FP32 even when the vision/reward forward uses AMP.
    preferred_scores = preferred_scores.float().reshape(-1)
    rejected_scores = rejected_scores.float().reshape(-1)
    if preferred_scores.shape != rejected_scores.shape:
        raise ValueError("preferred_scores and rejected_scores must have equal shape")
    logits = (preferred_scores - rejected_scores) / temperature
    return F.softplus(-logits).mean(), (logits > 0).float().mean()


class _ImageEncoder(nn.Module):
    """Small convolutional encoder retained only for CPU/unit-test debugging."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        width = max(feature_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, feature_dim, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class _DinoV3Encoder(nn.Module):
    """Frozen DINOv3 patch-token encoder with a spatial feature-map interface."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, __version__ as transformers_version
        except ImportError as exc:
            raise ImportError(
                "DINOv3 requires transformers>=4.56. Install it in the reward-model environment."
            ) from exc
        version = tuple(int(piece) for piece in transformers_version.split(".")[:2])
        if version < (4, 56):
            raise RuntimeError(
                f"DINOv3 requires transformers>=4.56; found {transformers_version}. "
                "Do not upgrade the VLA environment in place—use a separate reward-model environment."
            )
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.feature_dim = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "embed_dim", 384)))

    def train(self, mode: bool = True):
        # The reward head trains, but backbone dropout/stochastic layers must
        # remain disabled because it is deliberately frozen.
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        image_height, image_width = image.shape[-2:]
        if image_height % self.patch_size or image_width % self.patch_size:
            raise ValueError(f"DINOv3 inputs must be divisible by patch size {self.patch_size}")
        with torch.no_grad():
            output = self.model(pixel_values=image, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state
        patch_height, patch_width = image_height // self.patch_size, image_width // self.patch_size
        patch_count = patch_height * patch_width
        # DINOv3 prepends class/register tokens. Patch tokens occur last.
        if tokens.shape[1] < patch_count:
            raise RuntimeError("DINOv3 returned fewer tokens than its input patch grid")
        tokens = tokens[:, -patch_count:]
        return tokens.transpose(1, 2).reshape(image.shape[0], self.feature_dim, patch_height, patch_width)


def _as_batch(matrix: Tensor, batch_size: int, rows: int, cols: int, name: str) -> Tensor:
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.shape[-2:] != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols}) or (B, {rows}, {cols})")
    if matrix.shape[0] not in (1, batch_size):
        raise ValueError(f"{name} batch dimension must be 1 or match images")
    return matrix.expand(batch_size, -1, -1)


class TrajectoryAnchorRewardModel(nn.Module):
    """Score a candidate local trajectory from visual evidence along its path.

    Coordinates use CHOP's base frame: ``x`` forward, ``y`` lateral, and
    optional ``z`` up.  ``T_cam_from_base`` must map homogeneous base-frame
    points to the camera frame used by ``intrinsics``.  This calibration is
    required by design: it anchors visual attention to physically meaningful
    locations rather than asking the model to infer camera geometry from
    preference labels alone.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        hidden_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 2,
        num_samples: int = 4,
        max_offset_fraction: float = 0.12,
        max_waypoints: int = 32,
        vision_backbone: str = "dinov3",
        dinov3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if max_waypoints < 2:
            raise ValueError("max_waypoints must be at least two")
        self.num_samples = num_samples
        self.cross_attention_heads = num_heads
        self.max_offset_fraction = max_offset_fraction
        self.max_waypoints = max_waypoints
        if vision_backbone == "dinov3":
            self.image_encoder = _DinoV3Encoder(dinov3_model_name)
            feature_dim = self.image_encoder.feature_dim
        elif vision_backbone == "cnn":
            self.image_encoder = _ImageEncoder(feature_dim)
        else:
            raise ValueError("vision_backbone must be 'dinov3' or 'cnn'")
        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.waypoint_query_embedding = nn.Embedding(max_waypoints, hidden_dim)
        if feature_dim % self.cross_attention_heads:
            raise ValueError("feature_dim must be divisible by num_heads for deformable attention")
        self.visual_projection = nn.Linear(feature_dim, hidden_dim)
        # True multi-head deformable cross-attention: every anchor-query head
        # owns both a feature subspace and M learned reference-relative samples.
        self.value_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.offset_head = nn.Linear(hidden_dim, num_heads * num_samples * 2)
        self.weight_head = nn.Linear(hidden_dim, num_heads * num_samples)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _trajectory_features(path_points: Tensor) -> Tuple[Tensor, Tensor]:
        """Return 8-D waypoint features and a base-frame footprint direction."""
        if path_points.ndim != 3 or path_points.shape[-1] not in (2, 3):
            raise ValueError("path_points must be (B, K, 2) or (B, K, 3)")
        xy = path_points[..., :2]
        bsz, steps, _ = xy.shape
        if steps < 2:
            raise ValueError("at least two waypoints are required")
        delta = torch.diff(xy, dim=1, prepend=torch.zeros_like(xy[:, :1]))
        # The first waypoint's tangent is inferred from its successor.
        delta[:, 0] = xy[:, 1] - xy[:, 0]
        segment = delta.norm(dim=-1).clamp_min(1e-5)
        tangent = delta / segment.unsqueeze(-1)
        yaw = torch.atan2(tangent[..., 1], tangent[..., 0])
        arc = segment.cumsum(dim=1)
        time = torch.linspace(0, 1, steps, device=xy.device, dtype=xy.dtype).expand(bsz, -1)
        curvature = torch.diff(yaw, dim=1, prepend=yaw[:, :1])
        z = path_points[..., 2] if path_points.shape[-1] == 3 else torch.zeros_like(time)
        features = torch.stack((
            xy[..., 0], xy[..., 1], z, torch.sin(yaw), torch.cos(yaw),
            arc, time, curvature,
        ), dim=-1)
        # Normal points left of travel; sampling +/- normal covers robot width.
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        return features, normal

    @staticmethod
    def _project(points: Tensor, intrinsics: Tensor, t_cam_from_base: Tensor) -> Tuple[Tensor, Tensor]:
        """Project BxKx3 base points to normalized grid-sample coordinates."""
        batch_size, steps, _ = points.shape
        intrinsics = _as_batch(intrinsics, batch_size, 3, 3, "intrinsics").to(points)
        transform = _as_batch(t_cam_from_base, batch_size, 4, 4, "T_cam_from_base").to(points)
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        camera = torch.einsum("bij,bkj->bki", transform, homogeneous)[..., :3]
        depth = camera[..., 2]
        pixels = torch.einsum("bij,bkj->bki", intrinsics, camera)
        uv = pixels[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
        # uv is converted to a normalized grid after image dimensions are known.
        return uv, depth > 1e-4

    @staticmethod
    def _to_grid(uv: Tensor, image_height: int, image_width: int) -> Tensor:
        x = 2 * uv[..., 0] / max(image_width - 1, 1) - 1
        y = 2 * uv[..., 1] / max(image_height - 1, 1) - 1
        return torch.stack((x, y), dim=-1)

    def _sample_visual_features(
        self, feature_map: Tensor, query: Tensor, anchors: Tensor, waypoint_valid: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Multi-head deformable cross-attention around projected anchors."""
        batch_size, channels, _, _ = feature_map.shape
        _, steps, _ = anchors.shape
        heads, samples_per_head = self.cross_attention_heads, self.num_samples
        head_dim = channels // heads
        offsets = torch.tanh(self.offset_head(query)).view(
            batch_size, steps, heads, samples_per_head, 2
        )
        offsets = offsets * self.max_offset_fraction
        # Each head learns an independent set of points relative to the same
        # calibrated physical reference. This is the Deformable-DETR mechanism.
        grids = anchors.unsqueeze(2).unsqueeze(2) + offsets
        sampled = F.grid_sample(
            self.value_projection(feature_map), grids.view(batch_size, steps * heads * samples_per_head, 1, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(-1).transpose(1, 2).view(batch_size, steps, heads, samples_per_head, channels)
        # A head reads only its corresponding learned C/H feature subspace.
        sampled = torch.einsum(
            "bkhmjd,hj->bkhmd",
            sampled.view(batch_size, steps, heads, samples_per_head, heads, head_dim),
            torch.eye(heads, device=feature_map.device, dtype=feature_map.dtype),
        )
        weights = self.weight_head(query).view(batch_size, steps, heads, samples_per_head)
        weights = weights.masked_fill(~waypoint_valid[:, :, None, None], -1e4).softmax(dim=-1)
        visual = (sampled * weights.unsqueeze(-1)).sum(dim=3).reshape(batch_size, steps, channels)
        return visual, weights

    def encode_image(self, image: Tensor) -> Tensor:
        """Encode an image once for one or more candidate-path scores."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (B, 3, H, W)")
        return self.image_encoder(image)

    def forward(
        self,
        image: Optional[Tensor],
        path_points: Tensor,
        intrinsics: Tensor,
        t_cam_from_base: Tensor,
        waypoint_valid: Optional[Tensor] = None,
        encoded_image_features: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, dict[str, Tensor]]:
        """Return one scalar preference score per candidate trajectory.

        ``image`` is Bx3xHxW, and point units must match calibration units
        (meters for CHOP).  Higher scores mean more preferred.
        """
        if image is None and encoded_image_features is None:
            raise ValueError("provide image or encoded_image_features")
        if image is not None:
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("image must have shape (B, 3, H, W)")
            if image.shape[0] != path_points.shape[0]:
                raise ValueError("image and path_points batch dimensions must match")
            height, width = image.shape[-2:]
        else:
            if image_size is None:
                raise ValueError("image_size is required when using cached image features")
            height, width = image_size
        if encoded_image_features is not None and encoded_image_features.shape[0] != path_points.shape[0]:
            raise ValueError("encoded_image_features batch dimension must match path_points")
        features, _ = self._trajectory_features(path_points)
        geometry_query = self.geometry_encoder(features)
        xyz = path_points if path_points.shape[-1] == 3 else torch.cat((path_points, torch.zeros_like(path_points[..., :1])), dim=-1)
        if xyz.shape[1] > self.max_waypoints:
            raise ValueError(f"at most {self.max_waypoints} waypoints are supported")
        # Path points supply fixed, calibrated Deformable-DETR reference
        # anchors; they are not learned image-space boxes or masks.
        uv, visible = self._project(xyz, intrinsics, t_cam_from_base)
        anchor_grid = self._to_grid(uv, height, width)
        visual_anchor_in_image = visible & (anchor_grid.abs() <= 1).all(dim=-1)
        sequence_valid = torch.ones_like(visual_anchor_in_image)
        feature_map = self.encode_image(image) if encoded_image_features is None else encoded_image_features
        if waypoint_valid is not None:
            if waypoint_valid.shape != sequence_valid.shape:
                raise ValueError("waypoint_valid must have shape (B, K)")
            sequence_valid = waypoint_valid.bool()
        waypoint_ids = torch.arange(xyz.shape[1], device=xyz.device)
        query = geometry_query + self.waypoint_query_embedding(waypoint_ids).unsqueeze(0)
        # Out-of-frame points remain part of the trajectory. grid_sample gives
        # zero local visual evidence there, while learned offsets can still
        # reach nearby in-frame context.
        local_visual, sampling_weights = self._sample_visual_features(feature_map, query, anchor_grid, sequence_valid)
        tokens = query + self.visual_projection(local_visual)
        tokens = self.trajectory_encoder(tokens, src_key_padding_mask=~sequence_valid)
        pooled = (tokens * sequence_valid.unsqueeze(-1)).sum(dim=1) / sequence_valid.sum(dim=1, keepdim=True).clamp_min(1)
        score = self.reward_head(pooled).squeeze(-1)
        if not return_details:
            return score
        return score, {
            "anchor_grid": anchor_grid,
            "anchor_valid": sequence_valid,
            "visual_anchor_in_image": visual_anchor_in_image,
            "sampling_weights": sampling_weights,
        }
