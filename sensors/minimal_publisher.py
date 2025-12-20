import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
"""
https://answers.ros.org/question/359029/

requires python 3.8, a different conda env is created for this script.

"""

class MinimalPublisher(Node):
      def __init__(self):
         """
         0: laptop rgb
         2: laptop IR
         6: image + dots
         """
         super().__init__('minimal_publisher')
         self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
         timer_period = 0.5  # seconds
         self.timer = self.create_timer(timer_period, self.timer_callback)
         self.i = 0
         self.im_list = []
         # self.cv_image = cv2.imread('screenshot.png') ### an RGB image 
         self.cap = cv2.VideoCapture(6)
         self.bridge = CvBridge()

      def timer_callback(self):
         ret, frame = self.cap.read()

         if ret:
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame))
            self.get_logger().info(f'image size {frame.shape}')

         # self.publisher_.publish(self.bridge.cv2_to_imgmsg(np.array(self.cv_image)))

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()