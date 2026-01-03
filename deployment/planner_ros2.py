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

# Python generic imports
import time
import sys
import csv


class Config():
    # simulation parameters

    def __init__(self,intensity_publisher,intensity_publisher_high):

        self.intensity_pub = intensity_publisher
        self.intensity_pub_high = intensity_publisher_high

        self.max_speed = 0.7  # [m/s]
        self.min_speed = 0.0 #0.05  # [m/s]
        self.max_yawrate = 0.7 #0.6  # [rad/s]
        self.max_accel = 1  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        
        self.v_reso = 0.15 #0.20  # [m/s]
        self.yawrate_reso = 0.1 #0.10 #0.05  # [rad/s]
        
        self.dt = 0.5  # [s]
        self.predict_time = 1.8 #2.3 #1.5 #3.0 #1.5  # [s]
        
        self.to_goal_cost_gain = 18 #10 #10.0 # lower = detour
        self.veg_cost_gain = 1.6
        self.speed_cost_gain = 5 #10   # 0.1   # lower = faster
        self.obs_cost_gain = 0 #3.2     # lower z= fearless
        
        # self.robot_radius = 0.65 #0.6  # [m]
        # self.robot_radius = 1.5
        self.robot_radius = 0.33
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0
        # self.r = rclpy.Node().create_rate(20)

        self.collision_threshold = 0.3 # [m]
        # confidence = input("Enter Confidence Threshold : ")
        # self.conf_thresh = float(confidence)
        self.conf_thresh = 0.80   # Confidence Threshold

        # DWA output
        self.min_u = []

        self.stuck_status = False
        self.happend_once = False
        self.stuck_count = 0
        self.pursuing_safe_loc = False
        self.okay_locations = []
        self.stuck_locations = []


        # Costmap
        self.scale_percent = 300 # percent of original size
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05
        
        print("Initialized Costmap!")
        self.costmap_baselink = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_rgb = cv2.cvtColor(self.costmap_baselink,cv2.COLOR_GRAY2RGB)
        self.costmap_baselink_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_low = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.planning_costmap = np.zeros(self.costmap_shape, dtype=np.uint8)

        self.intensitymap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_baselink = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.obs_low_mid_high = np.argwhere(self.costmap_baselink_low > 150) # should be null set

        # For cost map clearing
        self.obs_consideration_thresh = 100
        self.height_thresh = 75 #150
        self.intensity_thresh = 180
        self.high_factor = 1
        self.alpha = 0.35
        self.br = CvBridge()

    # Callback for Odometry
    def assignOdomCoords(self, msg):
        # print("---------------Inside Odom Callback------------------------")

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w]) #uses the library from ros2, leads to errors
        roll,pitch,theta = R.from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w]).as_euler('xyz')

        self.th = theta

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z 
        # print("Robot's current velocities", [self.v_x, self.w_z])

    # Callback for goal from POZYX
    def target_callback(self, data):
        print("---------------Inside Goal Callback------------------------")

        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        print("r and theta:",data.linear.x, data.linear.y)
        
        # Goal wrt robot frame        
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)
        
        # print("Self odom:",self.x, self.y)
        # print("Goals wrt odom frame:", self.goalX, self.goalY)

        # If goal is published as x, y coordinates wrt odom uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y

        # Callback for local costmap from move_base and converting it wrt robot frame

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
                reached = False
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
                
class PlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_costmap')

        self.qos_profile = qos_profile = QoSProfile(
                                                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                                    history=QoSHistoryPolicy.KEEP_LAST,  
                                                    depth=10  
                                                    )

        self.qos_profile_intensity = qos_profile = QoSProfile(
                                                    reliability=QoSReliabilityPolicy.RELIABLE,
                                                    history=QoSHistoryPolicy.KEEP_LAST,  
                                                    depth=10  
                                                    )

        # Create the publisher
        self.intensity_publisher = self.create_publisher(
            sensorImage, 
            "/intensity_map", 
            self.qos_profile_intensity
        )

        # Create the publisher
        self.intensity_publisher_high = self.create_publisher(
            sensorImage, 
            "/intensity_map_high", 
            self.qos_profile_intensity
        )

        self.config = Config(self.intensity_publisher,self.intensity_publisher_high)
        self.obs = Obstacles()
        
        self.sub_odom = self.create_subscription(Odometry, '/odom_lidar', self.config.assignOdomCoords,self.qos_profile)        # self.sub_laser = self.create_subscription(LaserScan, '/j100_0707/sensors/lidar3d_0/scan', lambda msg: self.obs.assignObs(msg, self.config), self.qos_profile)
        self.sub_goal = self.create_subscription(Twist, '/target/position', self.config.target_callback, self.qos_profile)

        choice = input("Publish? 1 or 0")

        # ros2 topic pub -1 /target/position geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        
        if(int(choice) == 1):
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            print("Publishing to cmd_vel")
        else:
            self.pub = self.create_publisher(Twist, "/dont_publish", 1)
            print("Not publishing!")

        self.speed = Twist()
        self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])
        self.u = np.array([0.0, 0.0])
                
        # For on-field visualization
        self.plan_map_pub = self.create_publisher(sensorImage, "/planning_costmap", 10)
        self.viz_pub = self.create_publisher(sensorImage, "/viz_costmap", 10) 
        # self.intensity_pub = self.create_publisher(sensorImage, "/intensity_map", 10) 
        self.br = CvBridge()

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self.main_loop()

    def main_loop(self):
        self.config.stuck_status = False

        if self.config.goalX == 0.0006 and self.config.goalY == 0.0006:
            self.speed.linear.x = 0.0
            self.speed.angular.z = 0.0
            self.x = np.array([self.config.x, self.config.y, self.config.th, 0.0, 0.0])

        elif not atGoal(self.config, self.x):
            # if (config.stuck_status == True or config.pursuing_safe_loc == True):
            #     self.speed = recover(self.config, self.speed)
            # else:
            self.u = dwa_control(self.x, self.u, self.config, self.obs.obst)
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

# Model to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    x[3] = u[0]
    x[4] = u[1]

    return x

# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt # next sample

    return traj

# Calculate trajectory, costings, and return velocities to apply to robot
def calc_final_input(x, u, dw, config, ob):

    xinit = x[:]
    min_cost = 10000.0
    config.min_u = u
    config.min_u[0] = 0.0
    
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 150, 255)

    count = 0
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + config.v_reso/2, config.v_reso):
        for w in np.arange(dw[2], dw[3] + config.yawrate_reso/2, config.yawrate_reso):
            count = count + 1 
            
            traj = calc_trajectory(xinit, v, w, config)

            # calc costs with weighted gains
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3]) # end v should be as close to max_speed to have low cost
            
            traj_cm_col, traj_cm_row = transform_traj_to_costmap(config, traj)
            
            # veg_cost = config.veg_cost_gain * calc_veg_cost(traj, config)
            # veg_cost = config.veg_cost_gain * calc_veg_cost_v2(config, traj, traj_cm_col, traj_cm_row)
            ob_cost = config.obs_cost_gain * calc_obstacle_cost(traj, ob, config)
            print("calc_obstacle_cost: ", ob_cost)

            # final_cost = to_goal_cost + veg_cost
            # final_cost = to_goal_cost + veg_cost + speed_cost
            # final_cost = to_goal_cost*(1 + veg_cost)
            final_cost = to_goal_cost + ob_cost + speed_cost
            
            # print(count, "v,w = %.2f %.2f"% (v, w))
            # print("Goal cost = %.2f"% to_goal_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)
            # print("Goal cost = %.2f"% to_goal_cost, "speed_cost = %.2f"% speed_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)

            
            # config.costmap_rgb = draw_traj(config, traj, yellow)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                config.min_u = [v, w]

    # print("Robot's current velocities", [config.v_x, config.w_z])
    # traj = calc_trajectory(xinit, config.v_x, config.w_z, config) # This leads to buggy visualization

    traj = calc_trajectory(xinit, config.min_u[0], config.min_u[1], config)
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
    # print("min_u = %.2f %.2f"% (config.min_u[0], config.min_u[1]), "Goal cost = %.2f"% to_goal_cost, "Veg cost = %.2f"% veg_cost_min, "Min cost = %.2f"% min_cost)

    # Visualization
    # dim = (int(config.costmap_rgb.shape[1] * config.scale_percent / 100), \
    #  int(config.costmap_rgb.shape[0] * config.scale_percent / 100)) 
    # resized = cv2.resize(config.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
    
    # cv2.imshow('costmap_wrt_robot', resized)
    # cv2.imshow('costmap_baselink', config.costmap_baselink)
    # cv2.waitKey(3)
    
    return config.min_u

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

# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    # print("dx, dy", dx, dy)
    cost = math.sqrt(dx**2 + dy**2)
    print("Cost: ", cost)
    return cost

def transform_traj_to_costmap(config, traj):
    # Get trajectory points wrt robot
    traj_odom = traj[:,0:2]
    traj_rob_x = (traj_odom[:, 0] - config.x)*math.cos(config.th) + (traj_odom[:, 1] - config.y)*math.sin(config.th)
    traj_rob_y = -(traj_odom[:, 0] - config.x)*math.sin(config.th) + (traj_odom[:, 1] - config.y)*math.cos(config.th)
    traj_norm_x = (traj_rob_x/config.costmap_resolution).astype(int)
    traj_norm_y = (traj_rob_y/config.costmap_resolution).astype(int)

    # Get traj wrt costmap
    traj_cm_col = config.costmap_shape[0]/2 - traj_norm_y
    traj_cm_row = config.costmap_shape[0]/2 - traj_norm_x

    return traj_cm_col, traj_cm_row

# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob

# Begin DWA calculations
def dwa_control(x, u, config, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, dw, config, ob)
    # print(u)
    return u

# Determine whether the robot has reached its goal
def atGoal(config, x):
    # check at goal
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False

if __name__ == '__main__':
    
    rclpy.init()
    node = PlannerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # cv2.destroyAllWindows()