# !/usr/bin/env python3

# Author: Connor McGuile
# Latest author: Gershom Seneviratne
# Feel free to use in any way.

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
from numpy.lib.stride_tricks import as_strided

# Visualization Libraries
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image

# Message types
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, CompressedImage
from sensor_msgs.msg import Image as sensorImage
from scipy.spatial.transform import Rotation as R

from typing import List, Optional, Tuple

class LaserScanConfig:
    max_angle: float = 180.0        # degrees
    min_angle: float = -180.0       # degrees
    angle_increment: float = 0.5    # degrees
    range_min: float = 0.12         # meters
    range_max: float = 3.5          # meters

class RobotConfig():

    max_speed = 0.7        # [m/s]
    min_speed = 0.0        # [m/s]
    max_yawrate = 0.7      # [rad/s]
    max_accel = 1          # [m/s^2]
    max_dyawrate = 3.2     # [rad/s^2]

    v_reso = 0.05          # [m/s]
    yawrate_reso = 0.025   # [rad/s]

    dt = 0.025             # [s]
    predict_time = 0.5     # [s]

    to_goal_cost_gain = 18  # lower = detour
    speed_cost_gain = 5     # lower = faster
    obs_cost_gain = 0       # lower = fearless

    robot_radius = 0.33

class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.collision_status = False

    # Custom range implementation to loop over LaserScan degrees with
    # a step and include the final degree
    def myRange(self,start,end,step):
        i = start
        while i < end:
            yield i
            i += step
        yield end


    # Callback for LaserScan
    def assignObs(self, msg, config):

        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        self.obst = set()   # reset the obstacle set to only keep visible objects
        
        maxAngle = 360
        scanSkip = 1
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        self.collision_status = False
        for angle in self.myRange(0,deg-1,scanSkip):
            distance = msg.ranges[angle]

            if (distance < 0.05) and (not self.collision_status):
                self.collision_status = True
                # print("Collided")
                # reset_robot(reached)

            if(angleCount < (deg / (2*scanSkip))):
                # print("In negative angle zone")
                angleValueNeg += (anglePerSlot)
                scanTheta = (angleValueNeg - 180) * math.pi/180.0


            elif(angleCount>(deg / (2*scanSkip))):
                # print("In positive angle zone")
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            # only record obstacles that are within 4 metres away

            else:
                scanTheta = 0

            angleCount += 1

            if (distance < 4):
                # angle of obstacle wrt robot
                # angle/2.844 is to normalise the 512 degrees in real world
                # for simulation in Gazebo, use angle/4.0
                # laser from 0 to 180

                objTheta =  scanTheta + config.th

                # round coords to nearest 0.125m
                obsX = round((config.x + (distance * math.cos(abs(objTheta))))*8)/8
                # determine direction of Y coord
                
                if (objTheta < 0):
                    obsY = round((config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((config.y + (distance * math.sin(abs(objTheta))))*8)/8


                # add coords to set so as to only take unique obstacles
                self.obst.add((obsX,obsY))
        # print(self.obst)
                
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
        
        self.sub_odom = self.create_subscription(Odometry, '/odom_lidar', self.config.assignOdomCoords,self.qos_profile)        # self.sub_laser = self.create_subscription(LaserScan, '/j100_0707/sensors/lidar3d_0/scan', lambda msg: self.obs.assignObs(msg, self.config), self.qos_profile)
        self.sub_goal = self.create_subscription(Twist, '/target/position', self.config.target_callback, self.qos_profile)

        choice = input("Publish? 1 or 0")
        
        if(int(choice) == 1):
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            print("Publishing to cmd_vel")
        else:
            self.pub = self.create_publisher(Twist, "/dont_publish", 1)
            print("Not publishing!")

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

    # ------------ ROS callbacks ---------------

    def on_goal_cartesian_rf(self, msg):
        """
        Goals defined wrt robot frame in cartesian coordinates
            msg.linear.x: x
            msg.linear.y: y
        """
        self.goalX = msg.linear.x
        self.goalY = msg.linear.y

        if self.odom_assigned:
            self.goalX =  self.x + self.goalX*np.cos(self.yaw) - self.goalY*np.sin(self.yaw)
            self.goalY = self.y + self.goalX*np.sin(self.yaw) + self.goalY*np.cos(self.yaw)

    def on_goal_cartesian_wf(self, msg):
        """
        Goals defined wrt odom frame in cartesian coordinates
            msg.linear.x: x     (m)
            msg.linear.y: y     (m)
        """
        self.goalX = msg.linear.x
        self.goalY = msg.linear.y

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

    # Callback for Odometry
    def on_odom(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        roll,pitch,yaw = R.from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w]).as_euler('xyz')
        self.yaw = yaw

        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z 

    def atGoal(self):
        if np.linalg.norm(self.X[:2] - np.array([self.goalX, self.goalY])) <= self.config.robot_radius:
            return True
        return False

    def calc_dynamic_window(self):
        # Dynamic window from robot specification
        Vs = [self.config.min_speed, self.config.max_speed,
            -self.config.max_yawrate, self.config.max_yawrate]

        # Dynamic window from motion model
        Vd = [self.x[3] - self.config.max_accel * self.config.dt,
            self.x[3] + self.config.max_accel * self.config.dt,
            self.x[4] - self.config.max_dyawrate * self.config.dt,
            self.x[4] + self.config.max_dyawrate * self.config.dt]

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
    def calc_to_goal_cost(trajs, config):
        costs = np.linalg.norm(trajs[:, -1, 0:2] - np.array([config.goalX, config.goalY]), axis=1)
        return costs

    def calc_final_input(self, dw: list[float], ob: Optional[Obstacles] = None):

        self.config.min_u = self.u
        self.config.min_u[0] = 0.0

        trajs = []
        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1] + self.config.v_reso/2, self.config.v_reso):
            for w in np.arange(dw[2], dw[3] + self.config.yawrate_reso/2, self.config.yawrate_reso):
                
                traj = self.compute_trajectory(v, w)
                trajs.append(traj)

        trajs = np.array(trajs)
        # calc costs with weighted gains
        to_goal_costs = self.config.to_goal_cost_gain * self.calc_to_goal_cost(trajs)
        speed_costs = self.config.speed_cost_gain * np.abs(self.config.max_speed - trajs[:, -1, 3]) 
        ob_costs = self.config.obs_cost_gain * self.calc_obstacle_cost(trajs, ob, self.config)

        final_cost = to_goal_costs + ob_costs + speed_costs
        
        return 

    def dwa_control(self, ob: Optional[Obstacles] = None):
        dw = self.calc_dynamic_window()
        u = self.calc_final_input(dw, ob)   
        return u

    def main_loop(self):

        if not self.atGoal():

            self.u = self.dwa_control()
            self.x[0] = self.config.x
            self.x[1] = self.config.y
            self.x[2] = self.config.th
            self.x[3] = self.u[0]
            self.x[4] = self.u[1]
            self.speed.linear.x = self.x[3]
            self.speed.angular.z = self.x[4]
        else:
            self.get_logger().info("Goal reached!")
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
            self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])
        # print("Speed values :" + str(self.speed))
        self.pub.publish(self.speed)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self.main_loop()
    
    # def transform_traj_to_costmap(config, traj):
    #     # Get trajectory points wrt robot
    #     traj_odom = traj[:,0:2]
    #     traj_rob_x = (traj_odom[:, 0] - config.x)*math.cos(config.th) + (traj_odom[:, 1] - config.y)*math.sin(config.th)
    #     traj_rob_y = -(traj_odom[:, 0] - config.x)*math.sin(config.th) + (traj_odom[:, 1] - config.y)*math.cos(config.th)
    #     traj_norm_x = (traj_rob_x/config.costmap_resolution).astype(int)
    #     traj_norm_y = (traj_rob_y/config.costmap_resolution).astype(int)

    #     # Get traj wrt costmap
    #     traj_cm_col = config.costmap_shape[0]/2 - traj_norm_y
    #     traj_cm_row = config.costmap_shape[0]/2 - traj_norm_x

    #     return traj_cm_col, traj_cm_row

# Calculate obstacle cost inf: collision, 0:free
def calc_obstacle_cost(traj, ob, config):
    skip_n = 2
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            ox = i[0]
            oy = i[1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0 / minr

def calc_obstacle_cost_batched(trajs: np.ndarray, ob: Obstacles, config, skip_n: int = 2) -> np.ndarray:
    """
    Vectorized obstacle cost over a batch of trajectories.
    trajs: (M, T, 5) array of trajectories.
    Returns cost per trajectory: inf on collision, else 1/min_distance.
    """
    if ob is None or not ob.obst:
        return np.zeros(trajs.shape[0], dtype=float)

    obs = np.array(list(ob.obst), dtype=float)  # (N,2)
    traj_pts = trajs[:, ::skip_n, :2]           # (M, T/skip, 2)

    diff = traj_pts[:, :, None, :] - obs[None, None, :, :]  # (M, T, N, 2)
    dists = np.linalg.norm(diff, axis=-1)                  # (M, T, N)

    min_d = dists.min(axis=(1, 2))
    collided = dists.min(axis=2).min(axis=1) <= config.robot_radius

    costs = np.empty(trajs.shape[0], dtype=float)
    costs[collided] = np.inf
    costs[~collided] = 1.0 / np.maximum(min_d[~collided], 1e-6)
    return costs

def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob

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
