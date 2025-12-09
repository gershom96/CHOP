import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

import json
from pathlib import Path

from policy_sources.omnivla.prismatic.models.backbones.vision import ImageTransform
from policy_sources.visualnav_transformer.train.vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    to_local_coords_tensor
)

_DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "data" / "lora-data" / "train.json"


class ViNTChopDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        dataset_path: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        image_transform: ImageTransform,
        data_split_folder: str,
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining_pct: float,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
    ):
        """
        Main ViNT dataset class

        Args:
            image_root (string): Directory with all the image data
            dataset_path (string): Directory with json file with trajectory data
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            data_split_folder (string): Directory with data splits
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining_pct (float): Percentage of negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
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
        self.dataset_name = dataset_name

        self.trajectory_cache, self._image_cache = self._load_index(self.dataset_path)
        self.max_dist = max_dist

        if image_transform is None:
            self.image_transform = self._default_transform
        else:
            self.image_transform = image_transform

        self.learn_angle = learn_angle
        self.normalize = normalize

        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config

        self.data_config = all_data_config[self.dataset_name]
        self.to_tensor = transforms.ToTensor()

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing

        self.negative_mining_pct = negative_mining_pct
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.obs_type = obs_type
        self.goal_type = goal_type

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _load_index(self, path: Path) -> List[Dict[str, Any]]:
        raw = json.loads(path.read_text())
        image_cache: List[Optional[Any]] = []

        trajectory_cache = self._build_trajectory_cache(raw)
        print("Cached trajectories:", len(trajectory_cache))
        image_cache = self._build_image_cache(trajectory_cache)
        return trajectory_cache, image_cache
    
    def _build_trajectory_cache(self, json_raw: List[Dict[str, Any]]):
        trajectory_cache = []
        for bag_entry in json_raw:
            bag_name = bag_entry.get("bag")
            for sample in bag_entry.get("samples", []):
                item = dict()
                item.setdefault("bag", bag_name)
                item["timestamp"] = sample.get("timestamp")
                item["frame_idx"] = sample.get("frame_idx")

                if item["frame_idx"] < self.context_size * self.waypoint_spacing
                    continue  # not enough context
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
        cache_filename = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}.lmdb")

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

    def _get_goal(self, idx: int) -> Dict[str, Any]:

        if torch.rand(1).item() >= self.negative_mining_pct:
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
                path_dist = 0.2 * +self.max_dist

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
        else:
            goal_idx = np.random.randint(0, len(self.trajectory_cache))

        goal_position = self.trajectory_cache[goal_idx]["position"][:2]
        goal_yaw = self.trajectory_cache[goal_idx]["yaw"]
        image_path = str(self.image_root / self.trajectory_cache[goal_idx]["image_path"])
        goal_image = self._load_image(image_path)

        return {
            "goal_position": goal_position,
            "goal_yaw": goal_yaw,
            "goal_image": goal_image,
        }

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _build_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, image_path: str):
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buf = txn.get(image_path.encode())
            if buf is None:
                # handle missing key gracefully
                print(f"LMDB missing key {image_path}")
                return None
            return img_path_to_data(io.BytesIO(buf), self.image_size)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

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
            print( "metric_spacing:", metric_spacing )
            actions = actions / (metric_spacing * self.waypoint_spacing)
        return actions.float()

    def _compute_actions(self, action_data: Dict[str, torch.Tensor], 
                         current_pos: torch.Tensor, current_yaw: float, 
                         goal_pos: torch.Tensor, goal_yaw: float): 

        pos_actions = self._process_actions(action_data.get("path_0", {}))
        neg_actions = self._process_actions(action_data.get("path_1", {}))

        goal_pos = to_local_coords_tensor(goal_pos, current_pos, current_yaw)
        goal_yaw_loc = goal_yaw - current_yaw

        return pos_actions, neg_actions, goal_pos, goal_yaw_loc
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.trajectory_cache)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        sample = self.trajectory_cache[i]
        cur_pos = sample["position"][:2]                            # w.r.t. world frame
        cur_yaw = sample["yaw"]                                     # w.r.t. world frame
        image_path = str(self.image_root / sample["image_path"])
        cur_image = self._load_image(image_path)

        # f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        
        goal = self._get_goal(i)
        goal_pos = goal["goal_position"]
        goal_yaw = goal["goal_yaw"]
        goal_image = goal["goal_image"]

        f_goal, goal_time, goal_is_negative = 

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
