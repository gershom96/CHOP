import math
from typing import Any, Dict, Optional, Type, Callable, List, Tuple
from pathlib import Path
import json
import os
import tqdm
import io
import random

import lmdb
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
import yaml

from policy_sources.omnivla.prismatic.vla.action_tokenizer import ActionTokenizer
from policy_sources.omnivla.prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from policy_sources.omnivla.prismatic.models.backbones.llm.prompting import PromptBuilder
from policy_sources.omnivla.prismatic.models.backbones.vision import ImageTransform
from transformers import PreTrainedTokenizerBase

from policy_sources.visualnav_transformer.train.vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    to_local_coords_tensor
)


_DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "data" / "lora-data" / "train.json"

def _unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized CHW tensor back to [0,1] range for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


class OmniVLAChopDataset(torch.utils.data.Dataset):
    """
    Wraps OmniVLAChopDataset to emit the same keys/format as GNM_Dataset for finetuning.
    """

    def __init__(
        self,
        image_root: str,
        dataset_path: Optional[str],
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        prompt_builder_fn: Type[PromptBuilder],
        image_transform: ImageTransform,
        image_size: Tuple[int, int],
        dataset_name: str,
        data_split_type: str,
        data_split_folder: str,
        waypoint_spacing: int,
        learn_angle: bool,
        predict_stop_token: bool = True,
        max_dist: float = 30.0,
        modality_choices=(4, 5, 6),
        normalize: bool = True,
        create_image_cache: bool = True
    ) -> None:
        super().__init__()
        
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
        self.image_root = Path(image_root)
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        self.dataset_name = dataset_name
        self.create_image_cache = create_image_cache

        self.trajectory_cache, self._image_cache = self._load_index(self.dataset_path)
        self.max_dist = max_dist

        if image_transform is None:
            self.image_transform = self._default_transform
        else:
            self.image_transform = image_transform

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.modality_choices = modality_choices
        self.learn_angle = learn_angle
        self.normalize = normalize

        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config

        self.data_config = all_data_config[self.dataset_name]
        self.to_tensor = transforms.ToTensor()

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # self._build_caches()

    def _get_image_cache(self):
        # Opens LMDB lazily. When called inside a worker, this creates a worker-local env.
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self._image_cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048
            )
        return self._image_cache
    
    def _load_image(self, image_path: str):
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buf = txn.get(image_path.encode())
            if buf is None:
                # handle missing key gracefully
                print(f"LMDB missing key {image_path}")
                return img_path_to_data(image_path, self.image_size)
            return img_path_to_data(io.BytesIO(buf), self.image_size)
        except Exception as e:
            # print(f"Failed to load image {image_path}: {e}")
            return img_path_to_data(image_path, self.image_size)

    def _process_actions(self, path_data: Dict[str, torch.Tensor]):
        pts = path_data.get("points", torch.empty(0, 3, dtype=torch.float32))
        yaws = path_data.get("yaws", torch.empty(0, dtype=torch.float32))

        if pts.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        if yaws.dim() > 1:
            yaws = yaws.squeeze()
        
        assert pts.shape[0] == yaws.shape[0], "Points and yaws must have the same number of waypoints"
        actions = pts[:, :2]  # ego-frame waypoints (x,y)

        if self.learn_angle and yaws.numel() > 0:
            yaw_col = yaws.reshape(-1, 1)
            actions = torch.cat([actions, yaw_col], dim=1)

        if self.normalize:
            metric_spacing = self.data_config.get("metric_waypoint_spacing", 1.0)
            # print( "metric_spacing:", metric_spacing )
            actions[:, :2] /= (metric_spacing * self.waypoint_spacing)
        return actions.float()
        
    def _compute_actions(self, action_data: Dict[str, torch.Tensor], 
                         current_pos: torch.Tensor, current_yaw: float, 
                         goal_pos: torch.Tensor, goal_yaw: float): 

        pos_actions = self._process_actions(action_data.get("path_0", {}))
        neg_actions = self._process_actions(action_data.get("path_1", {}))

        goal_pos = to_local_coords_tensor(goal_pos, current_pos, current_yaw)

        if self.normalize:
            # Normalizing pos_actions and neg_actions already done in _process_actions
            metric_spacing = self.data_config.get("metric_waypoint_spacing", 1.0)
            goal_pos /= (metric_spacing * self.waypoint_spacing)
        goal_yaw_loc = goal_yaw - current_yaw

        return pos_actions, neg_actions, goal_pos, goal_yaw_loc
    
    def _load_index(self, path: Path) -> List[Dict[str, Any]]:
        raw = json.loads(path.read_text())
        image_cache: List[Optional[Any]] = []

        trajectory_cache = self._build_trajectory_cache(raw)
        print("Cached trajectories:", len(trajectory_cache))
        if self.create_image_cache:
            image_cache = self._build_image_cache(trajectory_cache)
        return trajectory_cache, image_cache

    def __len__(self) -> int:
        return len(self.trajectory_cache)
    
    def _build_trajectory_cache(self, json_raw: List[Dict[str, Any]]):
        trajectory_cache = []
        for bag_entry in json_raw:
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
                    item["yaw"] = torch.as_tensor(sample["yaw"], dtype=torch.float32)
                item["stop"] = bool(sample.get("stop", False))
                item["image_path"] = sample.get("image_path")

                trajectory_cache.append(item)
        return trajectory_cache

    def _build_image_cache(self, trajectory_cache: List[Dict[str, Any]], use_tqdm: bool = True):
        cache_filename = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb")
        print("LMDB cache filename:", cache_filename)
        # build the LMDB file if missing (write once)
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(trajectory_cache, disable=not use_tqdm, dynamic_ncols=True,
                                  desc=f"Building LMDB cache for {self.dataset_name}")
            
            env = lmdb.open(cache_filename, map_size=2**40)
            txn = env.begin(write=True)
            try:
                for i, frame in enumerate(tqdm_iterator):
                    rel_path = frame["image_path"]
                    image_path = str(self.image_root / rel_path)

                    with open(image_path, "rb") as f:
                        img_bytes = f.read()
                    txn.put(image_path.encode(), img_bytes)

                    # Commit every N images to avoid giant in-memory transactions
                    if (i + 1) % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                # final commit
                txn.commit()
            finally:
                env.close()

        self._image_cache_path = cache_filename
        self._image_cache = None
        return None

    # def _build_image_cache(self, trajectory_cache: List[Dict[str, Any]], use_tqdm: bool = True):
    #     cache_filename = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb")
        
    #     env = lmdb.open(cache_filename, map_size=2**40)  # existing file is fine
    #     with env.begin(write=True) as txn:
    #         tqdm_iterator = tqdm.tqdm(trajectory_cache, disable=not use_tqdm, dynamic_ncols=True, desc=f"Filling LMDB for {self.dataset_name}")
    #         for i, frame in enumerate(tqdm_iterator):
    #             image_path = str(self.image_root / frame["image_path"])
    #             key = image_path.encode()
    #             if txn.get(key) is not None:
    #                 continue  # already cached
    #             with open(image_path, "rb") as f:
    #                 txn.put(key, f.read())
    #             if (i + 1) % 1000 == 0:
    #                 txn.commit()
    #                 txn = env.begin(write=True)
    #         txn.commit()
    #     env.close()
    #     self._image_cache_path = cache_filename
    #     self._image_cache = None
    #     return None

    def _convert_path(self, path_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        def to_tensor(key: str, shape: int = 3) -> torch.Tensor:
            data = path_data.get(key) or []
            if not data:
                return torch.empty((0,) if shape == 1 else (0, shape), dtype=torch.float32)
            return torch.as_tensor(data, dtype=torch.float32).reshape(-1, shape)

        return {
            # paths are already resampled in preprocessing; just tensorize
            "points": to_tensor("points", 3),
            "yaws": to_tensor("yaws", 1),
            # "left_boundary": to_tensor("left_boundary"),
            # "right_boundary": to_tensor("right_boundary"),
        }

    def _get_goal(self, idx: int) -> Dict[str, Any]:
        p0 = self.trajectory_cache[idx]["path_0"]["points"]
        p1 = self.trajectory_cache[idx]["path_1"]["points"]

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
        bag_name = self.trajectory_cache[idx]["bag"]
        walked = 0.0
        prev_idx = idx
        goal_idx = idx

        # march forward only within this bag
        while True:
            next_idx = goal_idx + 1
            if (
                next_idx >= len(self.trajectory_cache)
                or self.trajectory_cache[next_idx]["bag"] != bag_name
            ):
                break  # reached end of this bag

            pos_prev = self.trajectory_cache[prev_idx]["position"]
            pos_next = self.trajectory_cache[next_idx]["position"]
            if pos_prev is None or pos_next is None:
                break

            step = float(torch.norm(pos_next - pos_prev))
            walked += step
            prev_idx = next_idx
            goal_idx = next_idx

            if walked >= goal_dist:
                break

        goal_position = self.trajectory_cache[goal_idx]["position"][:2]
        goal_yaw = self.trajectory_cache[goal_idx]["yaw"]
        image_path = str(self.image_root / self.trajectory_cache[goal_idx]["image_path"])
        goal_image = self._load_image(image_path)

        return {
            "goal_position": goal_position,
            "goal_yaw": goal_yaw,
            "goal_image": goal_image,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.trajectory_cache[idx]

        cur_pos = sample["position"][:2]                            # w.r.t. world frame
        cur_yaw = sample["yaw"]                                     # w.r.t. world frame
        image_path = str(self.image_root / sample["image_path"])
        cur_image = self._load_image(image_path)

        goal = self._get_goal(idx)

        goal_pos = goal["goal_position"]                            # w.r.t. world frame
        goal_yaw = goal["goal_yaw"]                                 # w.r.t. world frame
        goal_image = goal["goal_image"]

        cur_image_large = TF.resize(cur_image, (224, 224))
        goal_image_large = TF.resize(goal_image, (224, 224))

        pos_actions, neg_actions, goal_pos, goal_yaw_loc = self._compute_actions(sample, cur_pos, cur_yaw, goal_pos, goal_yaw)

        if self.learn_angle:
            pos_actions = calculate_sin_cos(pos_actions)
            neg_actions = calculate_sin_cos(neg_actions)

        ### Adapting OpenVLA stle ###
        actions = pos_actions

        # ViNT-style goal pose (x, y, cos, sin) in ego frame
        goal_pose_cos_sin = torch.stack(
            (
                goal_pos[0],
                goal_pos[1],
                torch.cos(goal_yaw_loc),
                torch.sin(goal_yaw_loc),
            )
        ).float()

        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"No language instruction"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        #print("check!!", labels.size(), input_ids.size())
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        dataset_name = "scand_a"
        
        if random.random() > 0.5:
            pixel_values = self.image_transform(to_pil_image(cur_image_large))
            pixel_values_g = self.image_transform(to_pil_image(goal_image_large))         
            cur_image_large = cur_image_large         
            goal_image = goal_image
            actions = actions
            goal_pose_cos_sin = goal_pose_cos_sin               
        else:
            pixel_values = self.image_transform(to_pil_image(cur_image_large).transpose(Image.FLIP_LEFT_RIGHT))
            pixel_values_g = self.image_transform(to_pil_image(goal_image_large).transpose(Image.FLIP_LEFT_RIGHT))         
            cur_image_large = torch.flip(cur_image_large, [2])
            goal_image = torch.flip(goal_image, [2])
            actions[:,1] = -actions[:,1]
            actions[:,3] = -actions[:,3]   
            neg_actions[:,1] = -neg_actions[:,1]
            neg_actions[:,3] = -neg_actions[:,3]
            goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
            goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]           
        
        distance = torch.norm(goal_pos).item()
        obj_pose_norm = torch.zeros(2)    # dummy
        # Set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"        
        modality_list = [4, 5, 6]   
        if distance <= 20:
            modality_id = random.choice(modality_list)
        else:
            modality_id = random.choice(modality_list[0:2]) #tdisntace is long --> no image only

        #action select 1.0: raw action, 0.0: MBRA synthetic action            
        action_select_mask = torch.tensor(1.0)           

        return dict(
            pixel_values=pixel_values, 
            pixel_values_goal=pixel_values_g, 
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name, 
            modality_id=modality_id,
            actions=torch.as_tensor(actions),
            neg_actions=torch.as_tensor(neg_actions),
            action_select_mask = action_select_mask,
            goal_pose=goal_pose_cos_sin, 
            obj_pose_norm=obj_pose_norm, 
            img_PIL=to_pil_image(cur_image_large),
            gimg_PIL=to_pil_image(goal_image),
            temp_dist=distance,
            lan_prompt="No language instruction"       
        )
