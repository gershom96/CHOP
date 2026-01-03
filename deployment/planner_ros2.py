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
# from tf_transformations import euler_from_quaternion

# Python generic imports
import time
import sys
import csv

# This import is added to allow setting autonomous mode wihtout CLI
# from ghost_manager_interfaces.srv import EnsureMode, SetParam

# This import is added to avoid backend issues with matplotlib

# OpenCV
# sys.path.remove('/opt/ros/noetic/lib/python2.7/dist-packages')

# sys.path.append('/opt/ros/noetic/lib/python2.7/dist-packages')


# ouster based odom
# run laser start script on the tablet

class DwaCostmapNode(Node):
    def __init__(self):
        super().__init__('dwa_costmap')

        # For setting to Autonomous Mode

        # self.mode_client = self.create_client(EnsureMode, 'ensure_mode')
        # while not self.mode_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('EnsureMode service not available, waiting again...')
        # self.mode_req = EnsureMode.Request()
        # self.param_client = self.create_client(SetParam, 'set_param')
        # while not self.mode_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('SetParam service not available, waiting again...')
        # self.param_req = SetParam.Request()

        # self.ensure_mode("control_mode", 170)
        # self.ensure_mode("action", 2) # walk mode

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
        
        self.sub_odom = self.create_subscription(Odometry, '/j100_0707/platform/odom/filtered', self.config.assignOdomCoords,self.qos_profile)
        self.sub_laser = self.create_subscription(LaserScan, '/j100_0707/sensors/lidar3d_0/scan', lambda msg: self.obs.assignObs(msg, self.config), self.qos_profile)
        self.sub_goal = self.create_subscription(Twist, '/target/position', self.config.target_callback, self.qos_profile)
        # self.sub_veg_classification = self.create_subscription(Float32MultiArray, '/vegetation_classes', self.classification_callback, self.qos_profile)
        # self.sub_costmap = self.create_subscription(OccupancyGrid, '/low/move_base/local_costmap/costmap', self.config.costmap_callback)

        choice = input("Publish? 1 or 0")

        # ros2 topic pub -1 /target/position geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        
        if(int(choice) == 1):
            self.pub = self.create_publisher(Twist, '/j100_0707/cmd_vel', 10)
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

        

    def set_param(self, param_name, value, planner=False):
        self.param_req.param.name = param_name
        self.param_req.param.val = value
        self.param_req.param.planner = planner
        future = self.param_client.call_async(self.param_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info('%s' % response.result_str)

    def ensure_mode(self, field_name, valdes):
        self.mode_req.field = field_name
        self.mode_req.valdes = valdes
        future = self.mode_client.call_async(self.mode_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info('%s' % response.result_str)

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

    

    def classification_callback(self, data):

        # print("Received classification results!")

        # Define grid cells belonging to each quadrant of the image
        # (col, row) convention
        # top_left = [(84, 0), (100, 49)]
        # top_right = [(100, 0), (117, 49)]
        # bottom_left = [(84, 49), (100, 82)]
        # bottom_right = [(100, 49), (117, 82)]

        # Q1 = np.array([(col, row) for col in range(84, 100+1) for row in range(0, 49+1)])
        # Q2 = np.array([(col, row) for col in range(100, 117+1) for row in range(0, 49+1)])
        # Q3 = np.array([(col, row) for col in range(84, 100+1) for row in range(49, 82+1)])
        # Q4 = np.array([(col, row) for col in range(100, 117+1) for row in range(49, 82+1)])

        # Sanity check for modifying costmap for navigation. THIS IS THE CORRECT CONVENTION.
        # Note: (row, col) convention is used for np array
        # self.config.costmap_baselink_low[Q1[:,1], Q1[:,0]] = 200
        # self.config.costmap_baselink_low[Q2[:,1], Q2[:,0]] = 255
        # self.config.costmap_baselink_low[Q3[:,1], Q3[:,0]] = 255
        # self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] = 200
        # cv2.imshow("Modified Costmap", self.config.costmap_baselink_low)
        # cv2.waitKey(3)

        # Sanity check for modifying costmap for visualization. THIS IS THE CORRECT CONVENTION.
        # cv2.rectangle(self.config.costmap_rgb, pt1=(84, 0), pt2=(100, 49), color=(0,255,0), thickness= 1) 
        # cv2.rectangle(self.config.costmap_rgb, pt1=(100, 0), pt2=(117, 49), color=(0,255,0), thickness= 1) 
        # cv2.rectangle(self.config.costmap_rgb, pt1=(84, 49), pt2=(100, 82), color=(255,0,0), thickness= 1) 
        # cv2.rectangle(self.config.costmap_rgb, pt1=(100, 49), pt2=(117, 82), color=(255,0,0), thickness= 1) 
        # dim = (int(self.config.costmap_rgb.shape[1] * self.config.scale_percent / 100), int(self.config.costmap_rgb.shape[0] * self.config.scale_percent / 100)) 
        # resized = cv2.resize(self.config.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap', resized)
        # cv2.waitKey(3)

        # Row values
        top = 20
        bottom = 100 # Must remain constant
        mid = int((top + bottom)/2)

        top_left = [(84, top), (95, mid)]
        top_center = [(95, top), (106, mid)]
        top_right = [(106, top), (117, mid)]
        
        bottom_left = [(84, mid), (95, bottom)]
        bottom_center = [(95, mid),(106, bottom)]
        bottom_right = [(106, mid), (117, bottom)]

        Q1 = np.array([(col, row) for col in range(84, 95+1) for row in range(top, mid+1)])
        Q2 = np.array([(col, row) for col in range(95, 106+1) for row in range(top, mid+1)])
        Q3 = np.array([(col, row) for col in range(106, 117+1) for row in range(top, mid+1)])
        Q4 = np.array([(col, row) for col in range(84, 95+1) for row in range(mid, bottom+1)])
        Q5 = np.array([(col, row) for col in range(95, 106+1) for row in range(mid, bottom+1)])
        Q6 = np.array([(col, row) for col in range(106, 117+1) for row in range(mid, bottom+1)])

        # Clear Costmap
        # NOTE: Modify this based on the actual data being published
        # 0 - non-pliable, 1 - pliable
        # Even indices correspond to class number, Odd indices correspond to distance
        veg1 = data.data[0]
        veg2 = data.data[2]
        veg3 = data.data[4]
        veg4 = data.data[6]
        veg5 = data.data[8]
        veg6 = data.data[10]
        conf1 = math.exp(-self.config.alpha * data.data[1])
        conf2 = math.exp(-self.config.alpha * data.data[3])
        conf3 = math.exp(-self.config.alpha * data.data[5])
        conf4 = math.exp(-self.config.alpha * data.data[7])
        conf5 = math.exp(-self.config.alpha * data.data[9])
        conf6 = math.exp(-self.config.alpha * data.data[11])
        # print(conf1, conf2, conf3, conf4)


        # Q1
        if (veg1 == 1 and conf1 >= self.config.conf_thresh):
            # self.config.costmap_baselink_low[Q1[:,1], Q1[:,0]] = (self.config.costmap_baselink_low[Q1[:,1], Q1[:,0]] * (1-conf1))
            # self.config.costmap_baselink_low[Q1[:,1], Q1[:,0]] = 0
            # self.config.costmap_rgb[Q1[:,1], Q1[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(84, top), pt2=(95, mid), color=(0,255,0), thickness= 1)

        else:
            # Don't clear costmap
            cv2.rectangle(self.config.costmap_rgb, pt1=(84, top), pt2=(95, mid), color=(0,0,255), thickness= 1)


        # Q2
        if (veg2 == 1 and conf2 >= self.config.conf_thresh):
            # self.config.costmap_baselink_low[Q2[:,1], Q2[:,0]] = (self.config.costmap_baselink_low[Q2[:,1], Q2[:,0]] * (1-conf2))
            # self.config.costmap_baselink_low[Q2[:,1], Q2[:,0]] = 0
            # self.config.costmap_rgb[Q2[:,1], Q2[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(95, top), pt2=(106, mid), color=(0,255,0), thickness= 1)
        else:
            cv2.rectangle(self.config.costmap_rgb, pt1=(95, top), pt2=(106, mid), color=(0,0,255), thickness= 1)


        # Q3
        if (veg3 == 1 and conf3 >= self.config.conf_thresh):
            # Clear cost map
            # self.config.costmap_baselink_low[Q3[:,1], Q3[:,0]] = (self.config.costmap_baselink_low[Q3[:,1], Q3[:,0]] * (1-conf3))
            # self.config.costmap_baselink_low[Q3[:,1], Q3[:,0]] = 0
            # self.config.costmap_rgb[Q3[:,1], Q3[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(106, top), pt2=(117, mid), color=(0,255,0), thickness= 1)
        else:
            cv2.rectangle(self.config.costmap_rgb, pt1=(106, top), pt2=(117, mid), color=(0,0,255), thickness= 1)


        # Q4
        if (veg4 == 1 and conf4 >= self.config.conf_thresh):
            # Clear cost map
            # self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] = (self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] * (1-conf4))
            #self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] = 0
            # self.config.costmap_rgb[Q4[:,1], Q4[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(84, mid), pt2=(95, bottom), color=(0,255,0), thickness= 1) 
        else:
            cv2.rectangle(self.config.costmap_rgb, pt1=(84, mid), pt2=(95, bottom), color=(0,0,255), thickness= 1) 

        
        # Q5
        if (veg5 == 1 and conf5 >= self.config.conf_thresh):
            # Clear cost map
            # self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] = (self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] * (1-conf4))
            #self.config.costmap_baselink_low[Q5[:,1], Q5[:,0]] = 0
            # self.config.costmap_rgb[Q4[:,1], Q4[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(95, mid), pt2=(106, bottom), color=(0,255,0), thickness= 1) 
        else:
            cv2.rectangle(self.config.costmap_rgb, pt1=(95, mid), pt2=(106, bottom), color=(0,0,255), thickness= 1)


        # Q6
        if (veg6 == 1 and conf6 >= self.config.conf_thresh):
            # Clear cost map
            # self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] = (self.config.costmap_baselink_low[Q4[:,1], Q4[:,0]] * (1-conf4))
            #self.config.costmap_baselink_low[Q6[:,1], Q6[:,0]] = 0
            # self.config.costmap_rgb[Q4[:,1], Q4[:,0], :] = 0
            cv2.rectangle(self.config.costmap_rgb, pt1=(106, mid), pt2=(117, bottom), color=(0,255,0), thickness= 1) 
        else:
            cv2.rectangle(self.config.costmap_rgb, pt1=(106, mid), pt2=(117, bottom), color=(0,0,255), thickness= 1)

        # self.config.costmap_baselink_low = self.config.costmap_baselink_low.astype('uint8')

        # self.plan_map_pub.publish(self.br.cv2_to_imgmsg(self.config.costmap_baselink_low, encoding="mono8"))
        # self.viz_pub.publish(self.br.cv2_to_imgmsg(self.config.costmap_rgb, encoding="bgr8"))

        # cv2.imshow("Modified Costmap", self.costmap_baselink_low)
        # dim = (int(self.costmap_rgb.shape[1] * self.scale_percent / 100), int(self.costmap_rgb.shape[0] * self.scale_percent / 100)) 
        # resized = cv2.resize(self.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap', resized)
        # cv2.waitKey(3)

class Config():
    # simulation parameters

    def __init__(self,intensity_publisher,intensity_publisher_high):

        self.intensity_pub = intensity_publisher
        self.intensity_pub_high = intensity_publisher_high
        
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        # self.max_speed = 0.65  # [m/s]
        # self.min_speed = 0.0  # [m/s]
        # self.max_yawrate = 0.4  # [rad/s]
        # self.max_accel = 1  # [m/ss]
        # self.max_dyawrate = 3.2  # [rad/ss]
        # self.v_reso = 0.10  # [m/s]
        # self.yawrate_reso = 0.10  # [rad/s]
        # self.dt = 0.5  # [s]
        # self.predict_time = 1.5  # [s]
        # self.to_goal_cost_gain = 1.2 #2.4 #lower = detour
        # self.speed_cost_gain = 0.1 #lower = faster
        # self.obs_cost_gain = 4.0 #3.2 #lower z= fearless
        # self.robot_radius = 0.6  # [m]
        # self.x = 0.0
        # self.y = 0.0
        # self.v_x = 0.0
        # self.w_z = 0.0
        # self.goalX = 0.0006
        # self.goalY = 0.0006
        # self.th = 0.0
        # self.r = rospy.Rate(20)

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

        # Lidar Intensity Map subscriber
        # self.low_costmap_callback = self.create_subscription(OccupancyGrid, "/intensity_grid", self.intensity_map_callback, self.qos_profile) 
        # self.sub_high_costmap = Node.create_subscription(OccupancyGrid, "/high/move_base/local_costmap/costmap", self.config.high_costmap_callback, self.qos_profile)
        # self.sub_mid_costmap = Node.create_subscription(OccupancyGrid, "/mid/move_base/local_costmap/costmap", self.config.mid_costmap_callback, self.qos_profile)
        # self.sub_low_costmap = Node.create_subscription(OccupancyGrid, "/low/move_base/local_costmap/costmap", self.config.low_costmap_callback, self.qos_profile)

    # Callback for Odometry
    def assignOdomCoords(self, msg):
        # print("---------------Inside Odom Callback------------------------")

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w]) #uses the library from ros2, leads to errors
        (roll,pitch,theta) = self.euler_from_quaternion (rot_q.x,rot_q.y,rot_q.z,rot_q.w) #uses the code in config class
        # print("Roll :",np.degrees(roll) , "Pitch:", np.degrees(pitch), "Theta :", np.degrees(theta))

        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.z, -rot_q.x, -rot_q.y, rot_q.w]) # used when lego-loam is used
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

    #Callback for lidar based intensity map
    def intensity_map_callback(self,data):
        # print("---------------Inside Intensity map Callback------------------------")
        intensitymap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        intensitymap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        intensitymap_2d = np.rot90(np.fliplr(intensitymap_2d), 1, (1, 0))

        im_image = Image.fromarray(np.uint8(intensitymap_2d))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_baselink_pil = im_image.rotate(-yaw_deg)
        self.intensitymap_baselink = np.array(im_baselink_pil)
        # print("Intensity map size before :", np.shape(self.intensitymap_baselink))

        kernel = np.ones((28,28),np.uint8)
        dilated_im = cv2.dilate(self.intensitymap_baselink,kernel,iterations =1)
        self.intensitymap_baselink_inflated = np.array(dilated_im)
        self.intensitymap_baselink_inflated = np.rot90(np.uint8(self.intensitymap_baselink_inflated),2)

        
        # self.config.intensitymap_baselink_inflated = np.array(self.config.pool2d(self.config.intensitymap_baselink, kernel_size=19, stride=1, padding=9, pool_mode='max'))
        # print("Intensity map size AFTER :", np.shape(self.intensitymap_baselink_inflated))

        # # im_image_inflated = np.rot90(np.uint8(self.config.intensitymap_baselink_inflated),2)

        # self.config.intensitymap_baselink_inflated = np.rot90(np.uint8(self.config.intensitymap_baselink_inflated),2)

        # print("Min and max of intensity map : ",np.max(self.intensitymap_baselink_inflated),np.min(self.intensitymap_baselink_inflated))

        #publish inflated intensity map as an image
        # image_inflated = Image.fromarray(self.config.intensitymap_baselink_inflated)

        # self.intensity_pub.publish(self.br.cv2_to_imgmsg(self.intensitymap_baselink_inflated, encoding="mono8"))

        # cv2.imshow('raw intensity map_wrt_robot', np.uint8(intensitymap_2d))
        # cv2.imshow('inflated intensity map_wrt_robot', im_image_inflated)
        # cv2.waitKey(3)

    def high_costmap_callback(self, data):

        # print("Received high local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))
        yaw_deg = 0 # Now cost map published wrt baselink
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_high = np.array(cm_baselink_pil)

        kernel = np.ones((32,32),np.uint8)
        dilated_im = cv2.dilate(self.costmap_baselink_high,kernel,iterations =1)
        self.costmap_baselink_high_inflated = np.array(dilated_im)
        self.costmap_baselink_high_inflated = np.rot90(np.uint8(self.costmap_baselink_high_inflated),2)

        # self.intensity_pub_high.publish(self.br.cv2_to_imgmsg(self.costmap_baselink_high_inflated, encoding="mono8"))


    def mid_costmap_callback(self, data):

        # print("Received mid local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))

        # yaw_deg = self.th*180/math.pi
        yaw_deg = 0
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_mid = np.array(cm_baselink_pil)


    def low_costmap_callback(self, data):

        # print("Received low local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))

        # yaw_deg = self.th*180/math.pi
        yaw_deg = 0 # check costmap_local.yaml. The frame has been changed from odom to base_link
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_low = np.array(cm_baselink_pil)
        self.costmap_rgb = cv2.cvtColor(self.costmap_baselink_low, cv2.COLOR_GRAY2RGB)

        kernel = np.ones((26,26),np.uint8)
        dilated_im = cv2.dilate(self.costmap_baselink_low,kernel,iterations =1)
        self.costmap_baselink_low = np.array(dilated_im)
        self.costmap_baselink_low = np.rot90(np.uint8(self.costmap_baselink_low),2)

        # Robot location on costmap
        rob_x = int(self.costmap_rgb.shape[0]/2)
        rob_y = int(self.costmap_rgb.shape[1]/2)

        # Visualization
        # Mark the robot on costmap 
        self.costmap_rgb = cv2.circle(self.costmap_rgb, (rob_x, rob_y), 4, (255, 0, 255), -1)
        self.costmap_sum()
        
        # dim = (int(self.costmap_baselink_low.shape[1] * self.scale_percent / 100), int(self.costmap_baselink_low.shape[0] * self.scale_percent / 100)) 
        # resized = cv2.resize(self.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap_wrt_robot', resized)
        # cv2.waitKey(3)

    def costmap_sum(self):
        costmap_sum = self.costmap_baselink_low + self.costmap_baselink_mid + self.costmap_baselink_high
        
        # self.obs_low_mid_high = np.argwhere(costmap_sum > self.height_thresh) # (returns row, col)

        # New Marking
        # self.obs_low_mid_high = np.argwhere(self.costmap_baselink_high > self.height_thresh)
        self.obs_low_mid_high = np.argwhere(self.intensitymap_baselink_inflated > self.intensity_thresh)

        if(self.obs_low_mid_high.shape[0] != 0):
            self.costmap_rgb = self.tall_obstacle_marker(self.costmap_rgb, self.obs_low_mid_high)
        else:
            pass

    def tall_obstacle_marker(self, rgb_image, centers):
        # Marking centers red = (0, 0, 255), or orange = (0, 150, 255)
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 0
        rgb_image[centers[:, 0], centers[:, 1], 2] = 255
        return rgb_image


    def euler_from_quaternion(self, x, y, z, w):
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

        return roll_x, pitch_y, yaw_z # in radians

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
    veg_cost_min = config.veg_cost_gain * calc_veg_cost(traj, config)
    # print("min_u = %.2f %.2f"% (config.min_u[0], config.min_u[1]), "Goal cost = %.2f"% to_goal_cost, "Veg cost = %.2f"% veg_cost_min, "Min cost = %.2f"% min_cost)
    config.costmap_rgb = draw_traj(config, traj, green)

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

def calc_veg_cost(traj, config):
    # print("Trajectory end-points wrt odom", traj[-1, 0], traj[-1, 1])

    # Convert traj points to robot frame
    x_end_odom = traj[-1, 0]
    y_end_odom = traj[-1, 1]

    # Trajectory approx mid-points
    x_mid_odom = traj[math.floor(len(traj)/2), 0]
    y_mid_odom = traj[math.floor(len(traj)/2), 1]

    x_end_rob = (x_end_odom - config.x)*math.cos(config.th) + (y_end_odom - config.y)*math.sin(config.th)
    y_end_rob = -(x_end_odom - config.x)*math.sin(config.th) + (y_end_odom - config.y)*math.cos(config.th)
    x_mid_rob = (x_mid_odom - config.x)*math.cos(config.th) + (y_mid_odom - config.y)*math.sin(config.th)
    y_mid_rob = -(x_mid_odom - config.x)*math.sin(config.th) + (y_mid_odom - config.y)*math.cos(config.th)


    # int() and floor() behave differently with -ve numbers. int() is symmetric. 
    # cm_col = config.costmap_shape[0]/2 - math.floor(y_end_rob/config.costmap_resolution)
    # cm_row = config.costmap_shape[1]/2 - math.floor(x_end_rob/config.costmap_resolution)
    cm_col = config.costmap_shape[0]/2 - int(y_end_rob/config.costmap_resolution)
    cm_row = config.costmap_shape[1]/2 - int(x_end_rob/config.costmap_resolution)

    cm_mid_col = config.costmap_shape[0]/2 - int(y_mid_rob/config.costmap_resolution)
    cm_mid_row = config.costmap_shape[1]/2 - int(x_mid_rob/config.costmap_resolution)


    # !!! NOTE !!!: IN COSTMAP, VALUES SHOULD BE ACCESSED AS (ROW,COL). FOR VIZ, IT SHOULD BE (COL, ROW)! 
    # Sanity Check: Drawing end and mid points
    # config.costmap_rgb = cv2.circle(config.costmap_rgb, (int(cm_col), int(cm_row)), 1, (255, 255, 255), 1)
    # config.costmap_rgb = cv2.circle(config.costmap_rgb, (int(cm_mid_col), int(cm_mid_row)), 1, (0, 255, 0), 1)
    
    # print("Value at end-point = ", config.costmap_baselink[int(cm_row), int(cm_col)])
    # print("Max and min of costmap: ", np.max(config.costmap_baselink), np.min(config.costmap_baselink))

    # Cost which only considers trajectory end point
    # veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)]
    
    # Cost which considers trajectory mid point and end point
    veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)] + config.costmap_baselink_low[int(cm_mid_row), int(cm_mid_col)]

    ## Adding height cost
    veg_cost = veg_cost + config.high_factor*config.costmap_baselink_high[int(cm_row), int(cm_col)] + config.high_factor*config.costmap_baselink_high[int(cm_mid_row), int(cm_mid_col)]
    # print("Added the height cost map here ....")

    # print("Costmap baselink low: ", np.shape(config.costmap_baselink_low))

    # print("veg cost :",veg_cost)

    # print("intensitymap_baselink_inflated: ", np.shape(config.intensitymap_baselink_inflated))

    #Adding intensity + height cost
    veg_cost = veg_cost + config.intensitymap_baselink_inflated[int(cm_row), int(cm_col)] + config.intensitymap_baselink_inflated[int(cm_mid_row), int(cm_mid_col)]
    # veg_cost = veg_cost + config.intensitymap_baselink[int(cm_row), int(cm_col)] + config.intensitymap_baselink[int(cm_mid_row), int(cm_mid_col)]
    # print("Added the intensity map here ....")

    # print("Sizes of the costmap and intesnity map :", np.shape(config.costmap_baselink_low), np.shape(config.intensitymap_baselink_inflated))

    return veg_cost

def calc_veg_cost_v2(config, traj, traj_cm_col, traj_cm_row):
    # NOTE: Planning costmap is set as glass_costmap_inflated
    config.planning_costmap = config.intensitymap_baselink_inflated
    
    
    # # print("Trajectory end-points wrt odom", traj[-1, 0], traj[-1, 1])

    # # Convert traj points to robot frame
    # x_end_odom = traj[-1, 0]
    # y_end_odom = traj[-1, 1]

    # # Trajectory approx mid-points
    # x_mid_odom = traj[int(math.floor(len(traj)/2)), 0]
    # y_mid_odom = traj[int(math.floor(len(traj)/2)), 1]

    # # print(x_end_odom, x_mid_odom)
    # # print("Odometry:", config.x, config.y)

    # x_end_rob = (x_end_odom - config.x)*math.cos(config.th) + (y_end_odom - config.y)*math.sin(config.th)
    # y_end_rob = -(x_end_odom - config.x)*math.sin(config.th) + (y_end_odom - config.y)*math.cos(config.th)
    # x_mid_rob = (x_mid_odom - config.x)*math.cos(config.th) + (y_mid_odom - config.y)*math.sin(config.th)
    # y_mid_rob = -(x_mid_odom - config.x)*math.sin(config.th) + (y_mid_odom - config.y)*math.cos(config.th)

    # # NOTE: int() and floor() behave differently with -ve numbers. int() is symmetric. 
    # cm_col = config.costmap_shape[0]/2 - int(y_end_rob/config.costmap_resolution)
    # cm_row = config.costmap_shape[1]/2 - int(x_end_rob/config.costmap_resolution)
    # cm_mid_col = config.costmap_shape[0]/2 - int(y_mid_rob/config.costmap_resolution)
    # cm_mid_row = config.costmap_shape[1]/2 - int(x_mid_rob/config.costmap_resolution)

    # # print("End point coordinates", cm_col, cm_row)
    # # print("Mid point coordinates", cm_mid_col, cm_mid_row)

    
    # print("traj_odom", traj_odom)
    # print(traj_rob_x, traj_rob_y)
    # print(traj_cm_row, traj_cm_col)

    # Calculate cost based on distance to obstacle
    ob = np.argwhere(config.planning_costmap > config.obs_consideration_thresh) # (row, col) format
    # print("Size of obstacle set:", ob.shape)
    skip_n = 3
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            o_row = i[0]
            o_col = i[1]
            dx = traj_cm_col[ii] - o_col
            dy = traj_cm_row[ii] - o_row

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r
    


    # -----------------------------------------------------------------------------------------------------------


    # NOTE: IN COSTMAP, VALUES SHOULD BE ACCESSED AS (ROW,COL). FOR VIZ, IT SHOULD BE (COL, ROW)! 
    # Sanity Check: Drawing end and mid points
    # config.viz_costmap = cv2.circle(config.viz_costmap, (int(cm_col), int(cm_row)), 1, (255, 255, 255), 1)
    # config.viz_costmap = cv2.circle(config.viz_costmap, (int(cm_mid_col), int(cm_mid_row)), 1, (0, 255, 0), 1)
    
    # print("Value at end-point = ", config.costmap_baselink[int(cm_row), int(cm_col)])
    # print("Max and min of costmap: ", np.max(config.costmap_baselink), np.min(config.costmap_baselink))

    # Cost which only considers trajectory end point
    # veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)]
    
    # # Cost which considers trajectory mid point and end point
    # obs_cost = config.glass_costmap_inflated[int(cm_row), int(cm_col)] + config.glass_costmap_inflated[int(cm_mid_row), int(cm_mid_col)]
    

    # return obs_cost
    # print("Obstacle cost: ", 1.0 / minr)
    return 1.0 / minr

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

def draw_traj(config, traj, color):
    traj_array = np.asarray(traj)
    x_odom_list = np.asarray(traj_array[:, 0])
    y_odom_list = np.asarray(traj_array[:, 1])

    # print(x_odom_list.shape)

    x_rob_list, y_rob_list = odom_to_robot(config, x_odom_list, y_odom_list)
    cm_col_list, cm_row_list = robot_to_costmap(config, x_rob_list, y_rob_list)

    costmap_traj_pts = np.array((cm_col_list.astype(int), cm_row_list.astype(int))).T
    # print(costmap_traj_pts) 

    costmap_traj_pts = costmap_traj_pts.reshape((-1, 1, 2))
    config.costmap_rgb = cv2.polylines(config.costmap_rgb, [costmap_traj_pts], False, color, 1)
    
    return config.costmap_rgb

# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob

def robot_to_costmap(config, x_rob, y_rob):

    costmap_shape_list_0 = [config.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
    costmap_shape_list_1 = [config.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

    y_list = [math.floor(y/config.costmap_resolution) for y in y_rob]
    x_list = [math.floor(x/config.costmap_resolution) for x in x_rob]

    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
    # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

    return cm_col, cm_row

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

def is_robot_stuck(config):

    # Condition for robot being stuck
    # NOTE: This condition may need to be changed to change in position or orientation
    
    # print("Robot's stuck locations: ", config.stuck_locations)
    # print("Robot's okay locations: ", config.okay_locations)
    # print("DWA Action: ", config.min_u)
    # print("Robot's current vel: ", config.v_x, config.w_z)
    
    if ((not config.pursuing_safe_loc) and (config.min_u != [0, 0] and config.min_u != []) and (abs(config.v_x) <= 0.05 and abs(config.w_z) <= 0.05)):
        config.stuck_count = config.stuck_count + 1
    else:
        config.stuck_count = 0

    if (config.stuck_count > 15):
        print("Robot could be stuck!")
        if (([math.floor(config.x), math.floor(config.y)] not in config.stuck_locations) and ([math.floor(config.x), math.floor(config.y)] not in config.okay_locations)): 
            # Stuck locations will only have integer coordinates. The "resolution" of the list is 1 meter.
            # Store stuck location
            config.stuck_locations.append([math.floor(config.x), math.floor(config.y)]) 

        return True # Stuck_status
    
    else:
        if (([math.floor(config.x), math.floor(config.y)] not in config.okay_locations) and ([math.floor(config.x), math.floor(config.y)] not in config.stuck_locations)): 
            # Okay locations will only have integer coordinates. The "resolution" of the list is 1 meter.
            # Store stuck location
            config.okay_locations.append([math.floor(config.x), math.floor(config.y)])

            # Experimental!
            # if (len(config.okay_locations) > 5 and config.happend_once == False):
            #     print("Collected 5 points. Stuck status = True!")
            #     config.happend_once = True
            #     return True

        return False

def recover(config, speed):

    config.pursuing_safe_loc = True

    x_odom = config.okay_locations[-2][0]
    y_odom = config.okay_locations[-2][1]

    # Convert the goal locations wrt robot frame. The error will simply be the goals.
    error_x = (x_odom - config.x)*math.cos(config.th) + (y_odom - config.y)*math.sin(config.th)
    error_y = -(x_odom - config.x)*math.sin(config.th) + (y_odom - config.y)*math.cos(config.th)

    print("(Recovery Point) --- (RobX, RobY) --- (Error X, Error Y) ")
    print(x_odom, y_odom, config.x, config.y, error_x, error_y)

    # Proportional gain
    k_p = 0.5
    vel_x = k_p * error_x
    vel_y = k_p * error_y

    # Note: This velocity assignment is for Spot cos it can move laterally
    # For a differential drive robot, use difference in angle and use it to compute w
    speed.linear.x = vel_x
    speed.linear.y = vel_y
    speed.angular.z = 0.0

    print(vel_x, vel_y)

    if (error_x < 0.5 and error_y < 0.5):
        print("Reached Safe Location!")
        config.pursuing_safe_loc = False
        config.stuck_status = False

        # Wait for 5 secs
        # time.sleep(5)

    return speed



if __name__ == '__main__':
    
    rclpy.init()
    node = DwaCostmapNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # cv2.destroyAllWindows()