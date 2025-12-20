from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import rosbag

from evaluation.base_evaluator import BaseEvaluator

class GoalDistanceEvaluator(BaseEvaluator):
    """
    Evaluates how far the model's final predicted waypoint is from the sampled goal.
    Only uses camera + odom topics
    """

    def __init__( self, output_path: str, scand: bool = True, model: str = "omnivla"):
        super().__init__( output_path, model )

        self._eval_name = "goal_distance"
        if scand:
            self._eval_name += "_scand"

        self._open_output_files()

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing goal distance for {self.bag_name}")
        distances = []

        for frame in tqdm(self.frames, desc=f"Eval {self._eval_name} {self.bag_name}", unit="frame"):
            if frame.goal_idx == -1:
                continue

            goal_frame = self.frames[frame.goal_idx]
            path_xy = frame.path_ft if finetuned else frame.path_bl

            if path_xy is None or len(path_xy) == 0:
                continue

            final_local = np.array(path_xy[-1])
            # Rotate from robot frame to world frame using current yaw, then translate.
            c, s = np.cos(frame.yaw), np.sin(frame.yaw)
            rot = np.array([[c, -s], [s, c]])
            final_global = frame.pos + rot @ final_local

            dist_to_goal = float(np.linalg.norm(final_global - goal_frame.pos))
            distances.append(dist_to_goal)

        return distances
