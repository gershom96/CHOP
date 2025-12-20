#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import math
import torch
import json

import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage

from sensor_msgs.msg import CompressedImage, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# from tf.transformations import euler_from_quaternion

from models.cont_policy import StochasticContinuousPolicy
from torchvision import transforms
from utils.trajectory_generation import trajectory_image


paths = {
    0: "./checkpoints/BC/TD3_BC-reward_model_0-2025-04-30_16-48-50_checkpoint_2499_2025-04-30_21-49-11.pt",
    1: "./checkpoints/BC/TD3_BC-reward_model_4-2025-04-30_15-41-23_checkpoint_2999_2025-05-01_00-12-07.pt",
    2: "./checkpoints/BC/TD3_BC-scand_her_dataset-2025-04-29_22-28-15_checkpoint_3049_2025-04-30_07-52-27.pt",
    3: "./checkpoints/IQL/IQL-reward_model_0-2025-04-30_16-12-39_checkpoint_4499_2025-05-01_00-42-56.pt",
    4: "./checkpoints/IQL/IQL-reward_model_4-2025-04-30_15-41-08_checkpoint_4499_2025-05-01_00-10-54.pt",
    5: "./checkpoints/IQL/IQL-scand_her_dataset-2025-04-29_22-11-16_checkpoint_3899_2025-04-30_07-53-45.pt",
}

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians  (counterclockwise)
    yaw is rotation around z in radians  (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in

class ActorNode(Node):

    def __init__(self, model_path, device, publish_cmd):
        super().__init__('policy_runner')

        self.policy = StochasticContinuousPolicy(activation="linear", state_dim=2).to(device)
        self.load_model_weights(model_path, "actor", device)
        self.policy.eval()

        self.device = device
        self.bridge = CvBridge()
        self.config = Config()

        self.qos_profile  = QoSProfile(
                                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                        history=QoSHistoryPolicy.KEEP_LAST,  
                                        depth=10  
                                    )

        self.sub_odom = self.create_subscription(Odometry, "/odom_lidar", self.assign_odom_coords, self.qos_profile)
        self.sub_goal = self.create_subscription(Twist, "/target/position", self.target_callback, self.qos_profile)
        self.sub_img = self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, self.qos_profile)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.dummy_cmd_pub = self.create_publisher(Twist, "/dummy_vel", 10)

        self.viz_pub = self.create_publisher(Image, "/viz_pub", 10)

        self.x = self.y = self.th = self.v_x = self.w_z = None
        self.odomReady = False
        self.goalX = self.goalY = None
        self.target_ready = False
        self.image = self.cv_image = None
        self.dGoal = self.hGoal = None
        self.goal_reached = False

        self.dino_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_stats = self.load_data_stats("train/data_stats.json")
        self.means = self.data_stats["means"]
        self.stds = self.data_stats["stds"]

        self.publish_cmd = publish_cmd
        self.visualize = True

    def assign_odom_coords(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.th = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z
        self.odomReady = True
        if self.target_ready:
            dx = self.goalX - self.x
            dy = self.goalY - self.y
            self.dGoal = np.sqrt(dx**2 + dy**2)
            goal_theta = normalize_angle(np.arctan2(dy, dx))
            self.hGoal = normalize_angle(goal_theta - self.th)

    def target_callback(self, msg):
        if self.odomReady:
            radius = msg.linear.x
            theta = math.radians(msg.linear.y)
            goalX_rob = radius * math.cos(theta)
            goalY_rob = radius * math.sin(theta)
            self.goalX = self.x + goalX_rob * math.cos(self.th) - goalY_rob * math.sin(self.th)
            self.goalY = self.y + goalX_rob * math.sin(self.th) + goalY_rob * math.cos(self.th)
            self.target_ready = True

    def image_callback(self, msg):

        # print("at Image callback")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = cv_image.shape
            min_dim = min(h, w)
            cropped = cv_image[(h - min_dim)//2:(h + min_dim)//2, (w - min_dim)//2:(w + min_dim)//2]
            resized = cv2.resize(cropped, (224, 224))
            self.cv_image = resized
            image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            self.image = self.dino_transform(image_pil).unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().warn(f"Image processing error: {e}")

    def load_model_weights(self, model_path, key, device):
        ckpt = torch.load(model_path, map_location=device)
        clean_sd = {k.replace("module.", ""): v for k, v in ckpt[key].items()}
        self.policy.load_state_dict(clean_sd, strict=False)

    def get_observation(self):

        # print(self.image is not None, self.dGoal, self.hGoal)
        if self.image is not None and self.dGoal is not None and self.hGoal is not None:
            return {
                "image": self.image,
                "goal_distance": torch.tensor([[self.normalize(self.dGoal, "goal_distance")]], dtype=torch.float32).to(self.device),
                "heading_error": torch.tensor([[self.normalize(self.hGoal, "heading_error")]], dtype=torch.float32).to(self.device),
            }
        return None

    def get_action(self, obs):
        with torch.no_grad():
            actions, dist = self.policy(obs)
            return torch.cat([dist["v"].loc, dist["omega"].loc], dim=-1)

    def load_data_stats(self, file_path):
        """Loads the stored normalization statistics (mean & std)."""
        with open(file_path, "r") as f:
            return json.load(f)

    def normalize(self, value, key):
        return (value - self.means[key]) / (self.stds[key] + 1e-8)

    def isAtGoal(self, threshold=1.0):
        return self.dGoal is not None and self.dGoal < threshold

    def step(self):
        obs = self.get_observation()
        v, w = 0,0


        if self.isAtGoal():
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.goal_reached = True
            v = 0
            w = 0

        elif obs and not self.goal_reached:

            print("Here")
            action = self.get_action(obs).squeeze()
            v = min(action[0].item(), 0.3)
            w = action[1].item()
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.get_logger().info(f"v: {v:.2f}, w: {w:.2f}, dGoal: {self.dGoal:.2f}")

            if self.publish_cmd:
                self.cmd_pub.publish(cmd)
            else: 
                self.dummy_cmd_pub.publish(cmd)

        # print(f"v: {v}, w: {w}")
        self.visualize_trajectory(v, w)


    def visualize_trajectory(self, reward_v, reward_omega):
        if self.cv_image is not None and self.visualize:
            image = self.cv_image.copy()
            traj_img = trajectory_image(
                self.config.K_viz, self.config.R, self.config.t,
                0.6, image.copy(), self.config.dt, 224, 224,
                image, [reward_v], [reward_omega], self.config.steps,
                self.config.Purple, mode="lines"
            )
            ros_image = self.bridge.cv2_to_imgmsg(traj_img, encoding="bgr8")
            self.viz_pub.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = paths[3]  # Update as needed

    publish_cmd = bool(input("Publish to motors (1/0): ").strip())
    actor_node = ActorNode(model_path, device, publish_cmd)

    # rate = actor_node.create_rate(10)

    while rclpy.ok():
        rclpy.spin_once(actor_node, timeout_sec=0.1)
        actor_node.step()
        # rate.sleep()

    actor_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()