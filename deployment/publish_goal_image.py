#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish an image goal for OmniVLA.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("goal1.jpg"),
        help="Path to a JPEG or PNG goal image (default: deployment/goal1.jpg).",
    )
    parser.add_argument("--topic", default="/goal/image/compressed")
    parser.add_argument("--wait-seconds", type=float, default=5.0)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Goal image not found: {image_path}")

    image_format = image_path.suffix.lower().lstrip(".")
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"jpeg", "png"}:
        raise ValueError(f"Expected a JPEG or PNG image, got: {image_path.suffix}")

    rclpy.init()
    node = Node("goal_image_publisher")
    publisher = node.create_publisher(CompressedImage, args.topic, 10)

    deadline = time.monotonic() + args.wait_seconds
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    msg = CompressedImage()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.format = image_format
    msg.data = image_path.read_bytes()
    publisher.publish(msg)
    node.get_logger().info(
        f"Published {image_path} ({len(msg.data)} bytes) to {args.topic}"
    )

    # Give DDS time to deliver the one-shot message before shutting down.
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
