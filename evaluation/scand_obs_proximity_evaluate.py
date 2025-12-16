import glob
import os
import json
import argparse
from typing import Optional
import yaml

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path

import rosbag
from sensor_msgs.msg import LaserScan
from evaluation.metrics.obs_proximity import min_clearance_to_obstacles_ls
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model
from policy_sources.omnivla.inference.run_omnivla_modified import Inference
from policy_sources.visualnav_transformer.deployment.src.utils import transform_images
from policy_sources.visualnav_transformer.train.vint_train.training.train_utils import get_action
from PIL import Image as PILImage

@dataclass
class FrameItem:
    frame_idx: int
    image: np.ndarray
    laserscan: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    pos: np.ndarray
    yaw: float
    cum_distance: float = 0.0
    goal_idx: int = -1

class InferenceConfig:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./omnivla-finetuned-cast"    
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
class ProximityEvaluator():
    def __init__(self, bag_dir: str, output_path: str, test_train_split_path: str, bags_to_skip: str,
                 model: str, fov_angle: float = 90.0, downsample_factor: int = 6, sample_goals: bool = True,
                 max_distance: float = 20.0, min_distance: float = 2.0):
        # super().__init__()
        self.bag_dir = bag_dir
        self.bridge = CvBridge()
        self.test_train_split_path = test_train_split_path
        self.bags_to_skip = bags_to_skip
        self.output_path = output_path
        self.downsample_factor = downsample_factor
        self.model_name = model
        
        self.sample_goals = sample_goals
        self.fov_angle = fov_angle
        self.max_distance = max_distance
        self.min_distance = min_distance

        self.frames: list[FrameItem] = []

        parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
        parser.add_argument(
            "--config",
            "-c",
            default=f"configs/chop_{self.model_name}_vnt.yaml",
            type=str,
            help="Path to the config file in train_config folder",
        )
        args = parser.parse_args()

        with open("configs/chop_default_vnt.yaml", "r") as f:
            default_config = yaml.safe_load(f)

        config = default_config

        with open(args.config, "r") as f:
            user_config = yaml.safe_load(f)

        config.update(user_config)
        self.config = config
        self.context_frames = self.config.get("context_size", 0)

    def load_model(self, finetuned: bool = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if finetuned:
            ckpt_path = self.config.get("finetuned_model_path")
        else:
            ckpt_path = self.config.get("pretrained_model_path")
        
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path

        model = None
        state_dict = None
        noise_scheduler = None

        if self.config.get("model_type") in {"vint", "gnm", "nomad"}:
            model = deployment_load_model(str(ckpt_path), self.config, device)

            if self.model_name == "nomad":
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config["num_diffusion_iters"],
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    prediction_type='epsilon'
                )

        elif self.model_name == "omnivla":
            model = Inference(save_dir="./inference",
                            ego_frame_mode=True,
                            save_images=False, 
                            radians=True,
                            vla_config=InferenceConfig())
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        if model is None:
            raise RuntimeError("Model failed to initialize.")

        model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model, noise_scheduler

    def run_inference(self, model, frame: FrameItem, goal_frame: FrameItem, noise_scheduler=None):
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_idx = max(0, frame.frame_idx - self.context_frames)
        context_imgs = [f.image[:, :, ::-1] for f in self.frames[start_idx:frame.frame_idx + 1]]  # BGR -> RGB
        context_pil = [PILImage.fromarray(img) for img in context_imgs]
        goal_pil = PILImage.fromarray(goal_frame.image[:, :, ::-1])

        if self.model_name in {"vint", "gnm"}:
            obs_tensor = transform_images(context_pil, self.config["image_size"])
            goal_tensor = transform_images(goal_pil, self.config["image_size"])
            obs_tensor = obs_tensor.to(device)
            goal_tensor = goal_tensor.to(device)
            with torch.no_grad():
                _, action_pred = model(obs_tensor, goal_tensor)
            path_xy = action_pred[0, :, :2].detach().cpu().numpy()
        elif self.model_name == "nomad":
            if noise_scheduler is None:
                raise RuntimeError("Noise scheduler required for NoMaD inference.")
            obs_images = transform_images(context_pil, self.config["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            goal_tensor = transform_images(goal_pil, self.config["image_size"], center_crop=False).to(device)
            mask = torch.zeros(1, device=device).long()

            obsgoal_cond = model('vision_encoder', obs_img=obs_images, goal_img=goal_tensor, input_goal_mask=mask)
            obs_cond = obsgoal_cond

            num_diffusion_iters = self.config["num_diffusion_iters"]
            noise_scheduler.set_timesteps(num_diffusion_iters)
            with torch.no_grad():
                noisy_action = torch.randn((1, self.config["len_traj_pred"], 2), device=device)
                naction = noisy_action
                for k in noise_scheduler.timesteps:
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = get_action(naction)
            path_xy = naction[0, :, :2].detach().cpu().numpy()
        elif self.model_name == "omnivla":
            cur_img = PILImage.fromarray(frame.image[:, :, ::-1]) #BGR to RGB
            cur_pos = frame.pos
            cur_yaw = frame.yaw

            goal_img = PILImage.fromarray(goal_frame.image[:, :, ::-1])
            goal_pos = goal_frame.pos
            goal_yaw = goal_frame.yaw

            model.update_current_state(cur_img, cur_pos, cur_yaw)
            model.update_goal(goal_image_PIL=goal_img, 
                                    goal_utm=goal_pos,
                                    goal_compass=goal_yaw, 
                                    lan_inst_prompt=None)
            model.run()
            path_xy = model.waypoints[:, :2] * model.metric_waypoint_spacing  # Convert to meters
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        return path_xy

    def process_laserscan_msg(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        range_min = float(msg.range_min)
        range_max = min(float(msg.range_max), self.max_distance)

        # Replace NaNs with inf
        ranges = np.nan_to_num(ranges, nan=np.inf)
        # Clip very large values
        ranges[ranges > range_max] = np.inf

        # Apply FOV mask (centered at 0 yaw)
        if self.fov_angle < 360.0:
            angles = angle_min + np.arange(len(ranges)) * angle_increment
            half_fov = np.deg2rad(self.fov_angle) / 2.0
            mask = (angles >= -half_fov) & (angles <= half_fov)
            ranges = np.where(mask, ranges, np.inf)

        return ranges, angle_min, angle_increment, range_min, range_max
    
    def process_odom(self, msg):
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        return pos, yaw
    
    def sample_goal_indices(self):
        goal_dist = np.random.uniform(self.min_distance, self.max_distance, size=len(self.frames))
        for i in range(self.context_frames, len(self.frames)-1):
            cur_dist = self.frames[i].cum_distance
            for j in range(i + 1, len(self.frames)):
                next_dist = self.frames[j].cum_distance
                if next_dist - cur_dist >= goal_dist[i]:
                    self.frames[i].goal_idx = j
                    break
            else:
                self.frames[i].goal_idx = j

    def process_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.frames = []

        print(f"\n=== Processing {self.bag_name} ===")

        if "Jackal" in self.bag_name:
            self.image_topic = "/camera/rgb/image_raw/compressed"
            self.laserscan_topic = "/velodyne_2dscan"
            self.odom_topic = "/jackal_velocity_controller/odom"
        elif "Spot" in self.bag_name:
            self.image_topic = "/image_raw/compressed"
            self.laserscan_topic = "/scan"
            self.odom_topic = "/odom"
        print(f"[INFO] Using image topic: {self.image_topic}")

        with rosbag.Bag(bag_path, "r") as bag:

            count = 0
            cam_frames = 0
            scan_data = None
            last_pos = None
            pos = None
            yaw = None
            cum_distance = 0.0
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.laserscan_topic, self.odom_topic])):
                if topic == self.image_topic:
                    if cam_frames % self.downsample_factor == 0:

                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        if scan_data is not None and pos is not None and yaw is not None:
                            if last_pos == None:
                                cum_distance = 0.0
                            else:
                                cum_distance += np.linalg.norm(pos - last_pos)
                            last_pos = pos
                            ranges, angle_min, angle_increment, range_min, range_max = scan_data
                            self.frames.append(
                                FrameItem(
                                    frame_idx=count,
                                    image=cv_image,
                                    laserscan=ranges,
                                    angle_min=angle_min,
                                    angle_increment=angle_increment,
                                    range_min=range_min,
                                    range_max=range_max,
                                    pos=pos,
                                    yaw=yaw,
                                    cum_distance=cum_distance,
                                )
                            )
                            count += 1
                    cam_frames += 1

                if topic == self.laserscan_topic:
                    scan_data = self.process_laserscan_msg(msg)

                if topic == self.odom_topic:
                    pos, yaw = self.process_odom(msg)

            print(f"[INFO] Processed {len(self.frames)} frames from {self.bag_name}")
            
            if self.sample_goals:
                self.sample_goal_indices()
                self.sample_goals = False

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing obstacle proximity for {self.bag_name}")
        min_clearances = []
        model, noise_scheduler = self.load_model(finetuned=finetuned)

        for i, frame in enumerate(self.frames):
            if frame.goal_idx == -1:
                continue
            
            goal_frame = self.frames[frame.goal_idx]
            path_xy = self.run_inference(model, frame, goal_frame, noise_scheduler=noise_scheduler)

            angle_max = frame.angle_min + (len(frame.laserscan) - 1) * frame.angle_increment
            min_clearance = min_clearance_to_obstacles_ls(
                path_xy=path_xy,
                laserscan=frame.laserscan,
                angle_increment=frame.angle_increment,
                angle_min=frame.angle_min,
                angle_max=angle_max,
                range_min=frame.range_min,
                range_max=frame.range_max,
            )

            if min_clearance != np.inf:
                min_clearances.append(min_clearance)
        return min_clearances
                
    def run(self):
        bag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))

        if not bag_files:
            print(f"[ERROR] No .bag files found in {self.bag_dir}")
            return
        
        with open(self.test_train_split_path, 'r') as f:
            test_train_bags = json.load(f)

        for bp in bag_files:
            bag_name = os.path.basename(bp)
            if test_train_bags.get(bag_name, "train") == "train":
                continue
            
            self.process_bag(bp)
            min_clearance_finetuned = self.analyze_bag(finetuned=True)
            min_clearance_pretrained = self.analyze_bag(finetuned=False)

            self.sample_goals = True

        print(f"\n[DONE] Annotations written to {self.output_path}")

if __name__ == "__main__":
    evaluator = ProximityEvaluator(
        bag_dir="/path/to/bags",
        output_path="/path/to/output.json",
        test_train_split_path="/path/to/split.json",
        bags_to_skip="",
        model="vint",
        fov_angle=90.0,
        downsample_factor=6,
        sample_goals=True,
        max_distance=20.0,
    )
    evaluator.run()
