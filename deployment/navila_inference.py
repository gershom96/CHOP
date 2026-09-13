#!/usr/bin/env python3
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NavilaCommand:
    action: str
    value: float = 0.0


def parse_navila_command(text: str) -> NavilaCommand:
    normalized = text.strip().lower()
    if re.search(r"\bstop\b", normalized):
        return NavilaCommand("stop")

    match = re.search(r"\bmove forward\s+(\d+(?:\.\d+)?)\s*cm\b", normalized)
    if match:
        return NavilaCommand("move_forward", float(match.group(1)) / 100.0)

    match = re.search(r"\bturn\s+(left|right)\s+(\d+(?:\.\d+)?)\s*degree(?:s)?\b", normalized)
    if match:
        sign = 1.0 if match.group(1) == "left" else -1.0
        return NavilaCommand("turn", sign * math.radians(float(match.group(2))))

    raise ValueError(f"Could not parse NaVILA command: {text!r}")


def navila_command_to_path(
    command: NavilaCommand,
    waypoint_spacing: float,
    turn_radius: float,
) -> np.ndarray:
    if command.action == "stop":
        return np.empty((0, 2), dtype=np.float32)

    if command.action == "move_forward":
        num_points = max(1, int(math.ceil(command.value / waypoint_spacing)))
        x = np.linspace(command.value / num_points, command.value, num_points)
        return np.column_stack((x, np.zeros_like(x))).astype(np.float32)

    if command.action == "turn":
        num_points = max(1, int(math.ceil(abs(command.value) / math.radians(15.0))))
        theta = np.linspace(command.value / num_points, command.value, num_points)
        x = turn_radius * np.sin(np.abs(theta))
        y = np.sign(theta) * turn_radius * (1.0 - np.cos(theta))
        return np.column_stack((x, y)).astype(np.float32)

    raise ValueError(f"Unsupported NaVILA action: {command.action}")


class NavilaInference:
    def __init__(
        self,
        repo_path: str,
        model_path: str,
        model_base: str = None,
        num_video_frames: int = 8,
        max_new_tokens: int = 32,
    ):
        import torch

        repo_path = os.path.abspath(os.path.expanduser(repo_path))
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"NaVILA repository not found: {repo_path}")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_name, model_base)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.image_processor = image_processor
        self.num_video_frames = int(getattr(model.config, "num_video_frames", num_video_frames))
        self.max_new_tokens = max_new_tokens
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.separator_style = SeparatorStyle
        self.conv_templates = conv_templates
        self.keywords_stopping_criteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token

    def infer(self, images: List[Image.Image], instruction: str) -> str:
        frames = self._sample_and_pad_images(images)
        image_tokens = "<image>\n" * (len(frames) - 1)
        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_tokens}, and current observation <image>\n. '
            f'Your assigned task is: "{instruction}" '
            "Analyze this series of images to decide your next action, which could be turning left or right "
            "by a specific degree, moving forward a certain distance, or stop if the task is completed."
        )

        conv = self.conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = self.process_images(frames, self.image_processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16
        )
        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != self.separator_style.TWO else conv.sep2
        stopping_criteria = self.keywords_stopping_criteria([stop_str], self.tokenizer, input_ids)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor.half(),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output

    def _sample_and_pad_images(self, images: List[Image.Image]) -> List[Image.Image]:
        frames = [image.convert("RGB") for image in images[-self.num_video_frames :]]
        if not frames:
            raise ValueError("NaVILA inference requires at least one image")

        width, height = frames[-1].size
        while len(frames) < self.num_video_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
        return frames
