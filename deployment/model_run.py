#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import threading
from typing import Optional
import os, sys, importlib
import yaml 

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
from cv_bridge import CvBridge

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model
from policy_sources.omnivla.inference.run_omnivla_modified import Inference
from policy_sources.visualnav_transformer.deployment.src.utils import transform_images
from policy_sources.visualnav_transformer.train.vint_train.training.train_utils import get_action

class InferenceConfigOriginal:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000
    vla_path: str = "./weights/omnivla-finetuned-cast"   
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

class InferenceConfigFinetuned:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./weights/omnivla-finetuned-chop"   
    resume_step: Optional[int] = 222500
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

@dataclass
class FrameItem:
    image: np.ndarray
    pos: Optional[np.ndarray] = None
    yaw: Optional[float] = None

@dataclass
class ContextFrame:
    image: np.ndarray

class ModelNode(Node):
    def __init__(self, config_path: Optional[str] = None, model_name: Optional[str] = None, finetuned: bool = False):
        super().__init__("model_node")

        self.qos_profile  = QoSProfile(
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST,  
                        depth=15  
                    )
        # ---------- Params ----------
        self.declare_parameter("path_frame_id", "base_link")  # semantic frame name
        self.declare_parameter("waypoint_spacing", 0.38)                 # spacing

        self.path_frame_id = self.get_parameter("path_frame_id").value
        self.waypoint_spacing = float(self.get_parameter("waypoint_spacing").value)
        self.config_path = config_path
        self.model_name = model_name
        self.finetuned = finetuned

        # ---------- State ----------
        self._lock = threading.Lock()
        self._started_sent = False

        # latest observation cache (set by callbacks)
        self._have_cur_img = False
        self._have_goal_img = False
        self._have_cur_pose = False
        self._have_goal_pose = False
        self._have_context = False

        self.config = self._load_config()
        self.context_update_period = self.config.get("context_update_period", 0.3)
        self._cv = threading.Condition(self._lock)

        self._dirty = False          # something changed since last inference
        self._shutdown = False
        self._inference_running = False  # acts as "busy"
         
        # ---------- ROS I/O ----------
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.pub_path = self.create_publisher(Path, "/path", 10)
        
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.sub_goal_img = self.create_subscription(CompressedImage, "/goal/image/compressed", self.on_goal_image, 10)
        self.sub_goal_pose = self.create_subscription(PoseStamped, "/goal/pose", self.on_goal_pose, 10)
        self.sub_nav = self.create_subscription(Empty, "/nav_cmd", self.on_nav_cmd, 10)
        self.context_timer = self.create_timer(self.context_update_period, self.update_context_from_current)
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()

        self.bridge = CvBridge()
        self.create_subscription(CompressedImage, "/camera/image/compressed", self.on_image, 10)

        self.get_logger().info(
            f"step_m={self.waypoint_spacing}, frame_id={self.path_frame_id}, config_path={self.config_path}"
        )

        self.vla_config = InferenceConfigOriginal()
        self.vla_config_finetuned = InferenceConfigFinetuned()
        self.model, self.noise_scheduler = self._load_model(finetuned=True, )

        self.cur_frame = FrameItem(
            image=None,
            pos=None,
            yaw=None
        )

        self.goal_frame = FrameItem(
            image=None,
            pos=None,
            yaw=None
        )

        self.context_frames = [ContextFrame(image=None) for _ in range(self.config.get("num_context_frames", 0))]
        self.goal_img_needed = self.config.get("need_goal_img", True)

    def _to_path_msg(self, path_xy: np.ndarray) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id  # semantic: "start frame"

        for x, y in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        return msg

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if self.model_name not in config:
            raise ValueError(f"Model {self.model_name} not found in config.")
        return config[self.model_name]

    def _load_model(self, finetuned: bool = True, ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        noise_scheduler = None
        
        if self.config.get("model_type") in {"vint", "gnm", "nomad"}:
            ckpt_path = self.config["chop_finetuned_path"] if finetuned else self.config["pretrained_model_path"]
            sys.modules["vint_train"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train")
            sys.modules["vint_train.models"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train.models")
            sys.modules["vint_train.models.vint"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train.models.vint")

            model = deployment_load_model(str(ckpt_path), self.config, device)

            if self.model_name == "nomad":
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config["num_diffusion_iters"],
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    prediction_type='epsilon'
                )

        elif self.model_name == "omnivla":
            if finetuned:
                vla_config = self.vla_config_finetuned
            else:
                vla_config = self.vla_config

            model = Inference(save_dir="./inference",
                            ego_frame_mode=True,
                            save_images=False, 
                            radians=True,
                            vla_config=vla_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        if model is None:
            raise RuntimeError("Model failed to initialize.")

        # Some wrappers (e.g., OmnivLA Inference) are not nn.Modules; guard attribute usage.
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        return model, noise_scheduler

    def _inference_worker(self):
        while True:
            with self._cv:
                # Wait until: shutdown OR (dirty and ready and not busy)
                self._cv.wait_for(lambda: self._shutdown or
                                        (self._dirty and self._ready_to_infer_locked() and not self._inference_running))

                if self._shutdown:
                    return

                # Claim work
                self._inference_running = True
                self._dirty = False

                # Snapshot inputs (shallow copies are fine; you can deep-copy later if needed)
                cur = FrameItem(
                    image=None if self.cur_frame.image is None else self.cur_frame.image.copy(),
                    pos=None if self.cur_frame.pos is None else self.cur_frame.pos.copy(),
                    yaw=self.cur_frame.yaw
                )
                goal = FrameItem(
                    image=None if self.goal_frame.image is None else self.goal_frame.image.copy(),
                    pos=None if self.goal_frame.pos is None else self.goal_frame.pos.copy(),
                    yaw=self.goal_frame.yaw
                )

                # Snapshot context images (optional; safe)
                ctx_imgs = [ContextFrame(image=cf.image.copy()) for cf in self.context_frames if cf.image is not None]

                model = self.model
                noise_scheduler = self.noise_scheduler

                # Publish /started once, when we actually start inferencing
                if not self._started_sent:
                    self._started_sent = True
                    self.pub_started.publish(Empty())
                    self.get_logger().info("Published /started (once).")

            # ---- Run inference outside lock ----
            try:
                path_xy = self.run_inference(model=model, cur_frame=cur, goal_frame=goal, context_frames=ctx_imgs, noise_scheduler=noise_scheduler)
            except Exception as e:
                self.get_logger().error(f"Inference failed: {repr(e)}")
                path_xy = None

            if path_xy is not None:
                try:
                    self.pub_path.publish(self._to_path_msg(path_xy))
                except Exception as e:
                    self.get_logger().error(f"Publishing /path failed: {repr(e)}")

            with self._cv:
                self._inference_running = False
                # If something became dirty while we were inferencing, loop will run again immediately

    def _trigger_inference(self):
        with self._cv:
            self._dirty = True
            self._cv.notify()
    
    def _ready_to_infer_locked(self) -> bool:

        if self.model_name in {"vint", "gnm", "nomad"}:
            # current+goal images must exist.
            if not (self._have_cur_img and self._have_goal_img and self.config.get("num_context_frames", 0) > 0 and not self._have_context):
                return False
        elif self.model_name == "omnivla":
            if not (self._have_cur_img and (self._have_goal_img or self._have_goal_pose)):
                return False
        return True
    # ---------------- callbacks ----------------
    def on_goal_image(self, msg: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self.goal_frame.image = img
            self._have_goal_img = True
        self._trigger_inference()

    def on_image(self, msg: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._have_cur_img = True
            self.cur_frame.image = img
        self._trigger_inference()

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        with self._lock:
            self.cur_frame.pos = np.array([p.x, p.y])
            self.cur_frame.yaw = yaw
            self._have_cur_pose = True
        self._trigger_inference()


    def on_goal_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        with self._lock:
            self.goal_frame.pos = np.array([p.x, p.y])
            self.goal_frame.yaw = yaw
            self._have_goal_pose = True
        self._trigger_inference()

    def on_nav_cmd(self, _msg: Empty):
        # Placeholder for navigation trigger; currently no-op
        return

    def update_context_from_current(self):
        with self._lock:
            if not self.context_frames:
                return
            if self.cur_frame.image is None:
                return
            self.context_frames.pop(0)
            self.context_frames.append(ContextFrame(image=self.cur_frame.image.copy()))
            self._have_context = all(cf.image is not None for cf in self.context_frames)

        self._trigger_inference()

    def run_inference(self, model, cur_frame, goal_frame, context_frames, noise_scheduler=None):
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        context_imgs = [f.image[:, :, ::-1] for f in context_frames]  # BGR -> RGB
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
            path_xy = path_xy * self.waypoint_spacing # Scale to meter To.do: make this configurable
        elif self.model_name == "omnivla":
            cur_img = PILImage.fromarray(cur_frame.image[:, :, ::-1]) #BGR to RGB
            cur_pos = cur_frame.pos
            cur_yaw = cur_frame.yaw

            goal_img = PILImage.fromarray(goal_frame.image[:, :, ::-1])
            goal_pos = goal_frame.pos
            goal_yaw = goal_frame.yaw

            model.update_current_state(cur_img, cur_pos, cur_yaw)
            model.update_goal(goal_image_PIL=goal_img, 
                                    goal_utm=goal_pos,
                                    goal_compass=goal_yaw, 
                                    lan_inst_prompt=None)
            model.run()
            waypoints = model.waypoints.reshape(-1, model.waypoints.shape[-1])
            path_xy = waypoints[:, :2] * self.waypoint_spacing  # Convert to meters
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Normalize to shape (N, 2) for downstream metrics
        path_xy = np.asarray(path_xy)
        if path_xy.ndim == 3:
            path_xy = path_xy.reshape(-1, path_xy.shape[-1])
        if path_xy.shape[-1] > 2:
            path_xy = path_xy[:, :2]

        return path_xy
    
    def destroy_node(self):
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Run the model node")
    parser.add_argument("-c", "--config", type=str, help="Path to config file", default="./configs/chop_inference_run.yaml")
    parser.add_argument("-m", "--model", type=str, help="Model name", default="omnivla")
    parser.add_argument("--finetuned", action="store_true", help="Use finetuned weights")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ModelNode(config_path=args.config, model_name=args.model, finetuned=args.finetuned)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
