# !/usr/bin/env python3

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance (1997).


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Numerical + CV libraries
import math
import numpy as np

# Message types
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R

from typing import List, Optional, Tuple

class LaserScanConfig:
    max_angle: float = 180.0        # degrees
    min_angle: float = -180.0       # degrees
    angle_increment: float = 0.5    # degrees
    range_min: float = 0.25         # meters
    range_max: float = 4.0          # meters
    scan_skip: int = 1              # idxs

class RobotConfig():

    max_speed = 1.0        # [m/s]
    min_speed = 0.0        # [m/s]
    max_yawrate = 0.175    # [rad/s]
    max_accel = 1          # [m/s^2]
    max_dyawrate = 3.2     # [rad/s^2]

    v_reso = 0.05          # [m/s]
    yawrate_reso = 0.02    # [rad/s]

    dt = 0.025             # [s]
    predict_time = 0.5     # [s]

    to_goal_cost_gain = 18  # lower = detour
    speed_cost_gain = 5     # lower = faster
    obs_cost_gain = 0       # lower = fearless

    robot_radius = 0.33

class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = np.zeros((0,2), dtype=float)
        self.collision_status = False
        self.laserscan_config = LaserScanConfig()
        self.obs_resolution = 0.05
        self.norm_factor = 1 / self.obs_resolution

    def assignObs(self, msg, config):
        """
        Vectorized obstacle extraction from LaserScan.
        Assumes config has x, y, th fields for the robot pose in world frame.
        """
        ranges = np.asarray(msg.ranges, dtype=float)
        deg = ranges.shape[0]
        self.obst = np.zeros((0,2), dtype=float)
        if deg == 0:
            return

        angle_per_slot = 2.0 * math.pi / deg
        angles = (np.arange(deg) - deg / 2.0) * angle_per_slot  # map to [-pi, pi]

        valid = np.isfinite(ranges) & (ranges < self.laserscan_config.range_max)
        if not valid.any():
            self.collision_status = False
            return

        dist_valid = ranges[valid]
        self.collision_status = bool((dist_valid < self.laserscan_config.range_min).any())

        th = getattr(config, "th", 0.0)
        x0 = getattr(config, "x", 0.0)
        y0 = getattr(config, "y", 0.0)

        obj_theta = angles[valid] + th
        obs_x = x0 + dist_valid * np.cos(obj_theta)
        obs_y = y0 + dist_valid * np.sin(obj_theta)

        obs = np.stack([np.round(obs_x * self.norm_factor) / self.norm_factor,
                        np.round(obs_y * self.norm_factor) / self.norm_factor], axis=1)
        obs_unique = np.unique(obs, axis=0)
        self.obst = obs_unique
 
class Planner(Node):
    def __init__(self):
        super().__init__('dwa_costmap')

        self.qos_profile = QoSProfile(  
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,  
            depth=10  
        )

        self.config = RobotConfig()
        self.obs = Obstacles()
        
        self.sub_odom = self.create_subscription(Odometry, '/odom_lidar', self.on_odom, self.qos_profile)        # self.sub_laser = self.create_subscription(LaserScan, '/j100_0707/sensors/lidar3d_0/scan', lambda msg: self.obs.assignObs(msg, self.config), self.qos_profile)
        self.sub_goal = self.create_subscription(Twist, '/target/position', self.on_goal_cartesian_rf, self.qos_profile)
        # self.sub_laser = self.create_subscription(LaserScan, '/scan', lambda msg: self.obs.assignObs(msg, self.config), self.qos_profile)

        choice = input("Publish? 1 or 0")
        
        if(int(choice) == 1):
            self.ctrl_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            print("Publishing to cmd_vel")
        else:
            self.ctrl_pub = self.create_publisher(Twist, "/dont_publish", 1)
            print("Not publishing!")
        self.req_goal_pub = self.create_publisher(Empty, "/req_goal", 10)

        self.x = None
        self.y = None
        self.yaw = None
        self.v_x = 0.0
        self.w_z = 0.0

        self.speed = Twist()

        # State space representation
        self.X = np.array([self.x, self.y, self.yaw, self.v_x, self.w_z])
        self.U = np.array([self.v_x, self.w_z])

        self.odom_assigned = False

        self.goalX = None
        self.goalY = None
        self._goal_req_sent = False

    # ------------ ROS callbacks ---------------

    def on_goal_cartesian_rf(self, msg):
        """
        Goals defined wrt robot frame in cartesian coordinates
            msg.linear.x: x
            msg.linear.y: y
        """
        gX = msg.linear.x
        gY = msg.linear.y

        if self.odom_assigned:
            self.goalX =  self.x + gX*np.cos(self.yaw) - gY*np.sin(self.yaw)
            self.goalY = self.y + gX*np.sin(self.yaw) + gY*np.cos(self.yaw)
            self._goal_req_sent = False

    def on_goal_cartesian_wf(self, msg):
        """
        Goals defined wrt odom frame in cartesian coordinates
            msg.linear.x: x     (m)
            msg.linear.y: y     (m)
        """
        self.goalX = msg.linear.x
        self.goalY = msg.linear.y
        self._goal_req_sent = False

    def on_goal_spherical_rf(self, msg):
        """
        Goals defined wrt robot frame in spherical coordinates
            msg.linear.x: radius    (m)
            msg.linear.y: theta     (deg)
        """
        radius = msg.linear.x # this will be r
        theta = np.deg2rad(msg.linear.y) # this will be theta

        # Goal wrt robot frame
        goalX_rob = radius * np.cos(theta)
        goalY_rob = radius * np.sin(theta)

        if self.odom_assigned:
            self.goalX =  self.x + goalX_rob*np.cos(self.yaw) - goalY_rob*np.sin(self.yaw)
            self.goalY = self.y + goalX_rob*np.sin(self.yaw) + goalY_rob*np.cos(self.yaw)
            self._goal_req_sent = False
    
    def on_goal_spherical_wf(self, msg):
        """
        Goals defined wrt world frame in spherical coordinates
            msg.linear.x: radius    (m)
            msg.linear.y: theta     (deg)
        """
        radius = msg.linear.x # this will be r
        theta = np.deg2rad(msg.linear.y) # this will be theta

        # Goal wrt robot frame
        self.goalX = radius * np.cos(theta)
        self.goalY = radius * np.sin(theta)
        self._goal_req_sent = False

    # Callback for Odometry
    def on_odom(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        roll,pitch,yaw = R.from_quat([rot_q.x, rot_q.y, rot_q.z, rot_q.w]).as_euler('xyz')
        self.yaw = yaw

        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z

        self.X = np.array([self.x, self.y, self.yaw, self.v_x, self.w_z])
        self.odom_assigned = True

    def atGoal(self):
        if not self.odom_assigned:
            return False
        elif self.goalX is None or self.goalY is None:
            return False
        elif np.linalg.norm(self.X[:2] - np.array([self.goalX, self.goalY])) <= self.config.robot_radius:
            return True
        return False

    def calc_dynamic_window(self):
        # Dynamic window from robot specification
        Vs = [self.config.min_speed, self.config.max_speed,
            -self.config.max_yawrate, self.config.max_yawrate]

        # Dynamic window from motion model
        Vd = [self.X[3] - self.config.max_accel * self.config.dt,
            self.X[3] + self.config.max_accel * self.config.dt,
            self.X[4] - self.config.max_dyawrate * self.config.dt,
            self.X[4] + self.config.max_dyawrate * self.config.dt]

        #  [vmin, vmax, yawrate min, yawrate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
                max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw
    
    def compute_trajectory(self, v, w):
        """
        Generate trajectory states in world frame for constant (v, w).
        Returns array of shape (N, 5): x, y, yaw, v, w.
        """
        steps = int(self.config.predict_time / self.config.dt) + 1
        t_vals = np.linspace(0.0, self.config.predict_time, steps)

        if abs(w) < 1e-6:
            # Straight line motion
            x_vals = v * t_vals
            y_vals = np.zeros_like(t_vals)
        else:
            R = v / w
            x_vals = R * np.sin(w * t_vals)
            y_vals = R * (1 - np.cos(w * t_vals))

        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        pts_rel = np.vstack([x_vals, y_vals])
        pts_w = (np.array([[c, -s], [s, c]]) @ pts_rel).T
        pts_w[:, 0] += self.x
        pts_w[:, 1] += self.y

        yaw_vals = self.yaw + w * t_vals
        v_vals = np.full_like(t_vals, v)
        w_vals = np.full_like(t_vals, w)

        trajectory = np.stack([pts_w[:, 0], pts_w[:, 1], yaw_vals, v_vals, w_vals], axis=-1)
        return trajectory

    # Calculate goal cost via Pythagorean distance to robot
    def calc_to_goal_cost(self, trajs):
        costs = np.linalg.norm(trajs[:, -1, 0:2] - np.array([self.goalX, self.goalY]), axis=1)
        return costs
    

    def calc_obstacle_cost(self, trajs: np.ndarray, skip_n: int = 2) -> np.ndarray:
        """
        obstacle cost over a batch of trajectories.
        trajs: (M, T, 5) array of trajectories.
        Returns cost per trajectory: inf on collision, else 1/min_distance.
        """
        if self.obs is None or len(self.obs.obst) == 0:
            return np.zeros(trajs.shape[0], dtype=float)

        obs = np.array(self.obs.obst, dtype=float)        # (N,2)
        traj_pts = trajs[:, ::skip_n, :2]           # (M, T/skip, 2)

        diff = traj_pts[:, :, None, :] - obs[None, None, :, :]  # (M, T, N, 2)
        dists = np.linalg.norm(diff, axis=-1)                  # (M, T, N)

        min_d = dists.min(axis=(1, 2))
        collided = min_d <= self.config.robot_radius

        costs = np.empty(trajs.shape[0], dtype=float)
        costs[collided] = np.inf
        costs[~collided] = 1.0 / np.maximum(min_d[~collided], 1e-6)
        return costs

    def calc_final_input(self, dw: list[float], ob: Optional[Obstacles] = None):

        trajs = []
        action_pairs = []
        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1] + self.config.v_reso/2, self.config.v_reso):
            for w in np.arange(dw[2], dw[3] + self.config.yawrate_reso/2, self.config.yawrate_reso):
                
                traj = self.compute_trajectory(v, w)
                trajs.append(traj)
                action_pairs.append((v, w))

        trajs = np.array(trajs)
        # calc costs with weighted gains
        to_goal_costs = self.config.to_goal_cost_gain * self.calc_to_goal_cost(trajs)
        speed_costs = self.config.speed_cost_gain * np.abs(self.config.max_speed - trajs[:, -1, 3]) 
        
        # ob_costs = self.config.obs_cost_gain * self.calc_obstacle_cost(trajs, ob, self.config)
        # final_cost = to_goal_costs + ob_costs + speed_costs
        final_cost = to_goal_costs + speed_costs

        if final_cost.size == 0:
            return np.array([0.0, 0.0])
        elif final_cost.min() == np.inf:
            return np.array([0.0, 0.0])
        
        return np.array(action_pairs[np.argmin(final_cost)])

    def dwa_control(self, ob: Optional[Obstacles] = None):
        dw = self.calc_dynamic_window()
        U = self.calc_final_input(dw, ob)   
        return U

    def main_loop(self):
        if self.odom_assigned:
            if not self.atGoal():

                self.U = self.dwa_control()
                self.X[0] = self.x
                self.X[1] = self.y
                self.X[2] = self.yaw
                self.X[3] = self.U[0]
                self.X[4] = self.U[1]
                self.speed.linear.x = self.X[3]
                self.speed.angular.z = self.X[4]
            else:
                self.get_logger().info("Goal reached!")
                if not self._goal_req_sent:
                    self.req_goal_pub.publish(Empty())
                    self._goal_req_sent = True
                self.speed.linear.x = 0.0
                self.speed.angular.z = 0.0
                self.X = np.array([self.x, self.y, self.yaw, 0.0, 0.0])
            # print("Speed values :" + str(self.speed))
            self.ctrl_pub.publish(self.speed)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self.main_loop()
    
if __name__ == '__main__':
    
    rclpy.init()
    node = Planner()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
