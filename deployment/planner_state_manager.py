#!/usr/bin/env python3
"""
ROS2 helper that maintains two tiers of state:
1) ingestion buffers kept fresh by camera/odom callbacks
2) goal buffers (pose or goal image) that the planner consumes when ready

The planner can poll `get_planner_input()` whenever it finishes an execution
cycle to fetch the latest image/odom/goal snapshot without blocking callbacks.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from evaluation.utils.loaders import load_config, load_model


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Compute yaw from quaternion (ROS standard ordering)."""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(t3, t4))

class PlannerConfig:
    def __init__(self):
        
        self.n_v = 5
        self.n_w = 5
        self.v_min = 0.175
        self.v_max = 1.6
        self.w_bound= 0.13125

        self.v_res = 0.3
        self.w_res = 0.04

        self.dt = 0.05
        self.steps = 60 # 2 seconds

        self.robot_width_meters = 0.8
        
        self.Red = (0, 0, 255)  # Red
        self.Green = (0, 255, 0)  # Green
        self.Blue = (255, 0, 0)  # Blue
        self.Orange = (0, 123, 255)  # Orange
        self.Yellow = (3, 240, 252)
        self.Purple = (252, 3, 94)  # purple
        self.Pink = (235, 3, 252)

        # Initialize last action
        self.mask_size = 16

        self.fx = 613 * (self.mask_size/224)
        self.fy = 613 * (self.mask_size/224)
        self.cx = self.mask_size//2
        self.cy = self.mask_size//2
        
        # Camera matrix
        self.K = np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]])
        # Camera rotaion + translation
        self.R = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])
        self.t = np.array([[0], [-0.1], [0]])  # Translation vector

        self.fx_viz = 613 
        self.fy_viz = 613
        self.cx_viz = 112
        self.cy_viz = 112
        
        # Camera matrix
        self.K_viz = np.array([[self.fx_viz, 0, self.cx_viz],
                        [0, self.fy_viz, self.cy_viz],
                        [0, 0, 1]])

class Planner:

    def __init__(self, model_name: str = "omnivla", finetuned: bool = True):
        self.config = load_config(model_name, finetuned)
        self.model = load_model(self.config)
        self.planner_config = PlannerConfig()

    def update_goal(self):
        with self._lock:
            self._buffers.goal_xy = self._buffers.position_xy
            self._buffers.goal_image_bgr = self._buffers.image_bgr

    def plan_step(self, planner_input: 'PlannerBuffers') -> Tuple[float, float]:
        """
        Given the current observation and goal from the buffers,
        run the planner model to produce a (v, w) command.
        """

        self.update_goal()

    

@dataclass
class PlannerBuffers:
    image_bgr: Optional[np.ndarray] = None
    image_stamp: Optional[float] = None

    position_xy: Optional[Tuple[float, float]] = None
    yaw: Optional[float] = None
    odom_stamp: Optional[float] = None

    goal_xy: Optional[Tuple[float, float]] = None
    goal_image_bgr: Optional[np.ndarray] = None

    def ready_for_plan(self) -> bool:
        return (
            self.image_bgr is not None
            and self.position_xy is not None
            and (self.goal_xy is not None or self.goal_image_bgr is not None)
        )
    
class PlannerDataHolder(Node):
    """
    Keeps latest camera frame and odom in memory, plus a goal pose or goal image.
    Planner can pull snapshots as needed; callbacks never block on planning.
    """

    def __init__(
        self,
        camera_topic: str = "/camera/color/image_raw",
        odom_topic: str = "/odom",
        goal_topic: str = "/goal_pose",
    ):
        super().__init__("planner_coordinator")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._buffers = PlannerBuffers()
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self.create_subscription(Image, camera_topic, self._cb_image, qos)
        self.create_subscription(Odometry, odom_topic, self._cb_odom, qos)
        # Optional: accept geometry_msgs/PoseStamped goals (e.g., from RViz 2D Nav Goal)
        self.create_subscription(PoseStamped, goal_topic, self._cb_goal_pose, qos)

        self.get_logger().info(
            f"Listening: image={camera_topic}, odom={odom_topic}, goal={goal_topic}"
        )

    # --- Callbacks ---
    def _cb_image(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return
        with self._lock:
            self._buffers.image_bgr = cv_img
            self._buffers.image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _cb_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        with self._lock:
            self._buffers.position_xy = (pos.x, pos.y)
            self._buffers.yaw = yaw
            self._buffers.odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _cb_goal_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        with self._lock:
            self._buffers.goal_xy = (p.x, p.y)
            # Keep goal_image_bgr as-is; user may set via API.
        self.get_logger().info(f"Received goal pose ({p.x:.2f}, {p.y:.2f})")

    # --- External API for planner or higher-level orchestrator ---
    def set_goal_xy(self, x: float, y: float) -> None:
        with self._lock:
            self._buffers.goal_xy = (x, y)
        self.get_logger().info(f"Set goal (x, y)=({x:.2f}, {y:.2f})")

    def set_goal_image_bgr(self, img: np.ndarray) -> None:
        with self._lock:
            self._buffers.goal_image_bgr = img
        self.get_logger().info("Set goal image (BGR)")

    def get_planner_input(self) -> Tuple[PlannerBuffers, bool]:
        """
        Returns a copy of the current buffers and a readiness flag.
        Planner calls this after finishing an execution to start the next plan.
        """
        with self._lock:
            snapshot = PlannerBuffers(
                image_bgr=None if self._buffers.image_bgr is None else self._buffers.image_bgr.copy(),
                image_stamp=self._buffers.image_stamp,
                position_xy=self._buffers.position_xy,
                yaw=self._buffers.yaw,
                odom_stamp=self._buffers.odom_stamp,
                goal_xy=self._buffers.goal_xy,
                goal_image_bgr=None
                if self._buffers.goal_image_bgr is None
                else self._buffers.goal_image_bgr.copy(),
            )
        return snapshot, snapshot.ready_for_plan()


def main() -> None:
    rclpy.init()
    node = PlannerDataHolder()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            # Example planner hook
            snapshot, ready = node.get_planner_input()
            if not ready:
                continue
            # TODO: plug in your planner here.
            # Example: print state to show flow.
            node.get_logger().debug(
                f"Planner tick: pose={snapshot.position_xy}, yaw={snapshot.yaw}, "
                f"goal={snapshot.goal_xy}, image_ts={snapshot.image_stamp}"
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
