import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import pyrealsense2 as rs
import cv2
import numpy as np
import argparse


class CustomPublisher(Node):
    """
    Docstring for CustomPublisher
    - args.raw_img: we will publish the raw img
    - args.compress_img: we will publish the compressed img
    
    the depth image is let unused at the moment but can be extracted from:
    depth_image = np.asanyarray(depth_frame.get_data())

    references:
    https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    https://answers.ros.org/question/359029/
     - may require python3.8 to play nice with foxy
    """
    def __init__(self, args):
        super().__init__('custom_publisher')

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        # print(device.sensors)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        self.pipeline = pipeline
        if args.compress_img:
            self.publisher_compressed = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 10)
        if args.raw_img:
            self.publisher_normal = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    pass
                else:
                    # Convert images to numpy arrays
                    # depth_image = np.asanyarray(depth_frame.get_data())
                    frame = np.asanyarray(color_frame.get_data())
                    
                    if args.compress_img:
                        format = "jpg"
                        img_encode = cv2.imencode('.jpg', frame)[1]
                        byte_encode = np.array(img_encode).tobytes()

                        comp_img = CompressedImage()
                        comp_img.header.stamp = self.get_clock().now().to_msg()
                        comp_img.format = format
                        comp_img.data = byte_encode

                        self.publisher_compressed.publish(comp_img)
                    if args.raw_img:
                        self.publisher_normal.publish(self.bridge.cv2_to_imgmsg(frame))
                    
                    self.get_logger().info(f'image size {frame.shape}')

        finally:

            # Stop streaming
            self.pipeline.stop()
        


def main():
    parser = argparse.ArgumentParser(description="realsense publisher")
    parser.add_argument("--disable_raw", action="store_false", dest="raw_img", help="Disable publishing raw images")
    parser.add_argument("--disable_compress", action="store_false", dest="compress_img", help="Disable publishing compressed images")
    args = parser.parse_args()
    print("args:", args)
    rclpy.init(args=None)
    custom_publisher = CustomPublisher(args)
    rclpy.spin(custom_publisher)
    custom_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()