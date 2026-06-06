import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# BASE_DIR = Path(__file__).resolve().parent


class DataCollectionNode(Node):
    def __init__(self):
        super().__init__("data_collection_node")

        self.bridge = CvBridge()
        self.save_dir = Path(
            "C:\\Users\\tedjh\\Documents\\ml_projects\\robotic_car\\training_data"
        )
        self.get_logger().info(f"Saving data to {self.save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Open CSV to log controls alongside image filenames
        self.csv_file = open(f"{self.save_dir}/labels.csv", "a")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["image_filename", "state"])

        # Synchronised subscribers
        image_sub = message_filters.Subscriber(self, Image, "/camera/image_raw")
        cmd_sub = message_filters.Subscriber(self, String, "velocity")

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, cmd_sub], queue_size=10, slop=0.1
        )  # 100ms tolerance for matching messages
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, image_msg, cmd_msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        # Save image with timestamp as filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        cv2.imwrite(f"{self.save_dir}/{filename}", frame)

        # Log controls
        self.csv_writer.writerow([filename, cmd_msg.data])

        self.get_logger().info(f"Saved {filename}")


def main(args=None):
    with rclpy.init(args=args):
        node = DataCollectionNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()


if __name__ == "__main__":
    main()
