import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture("rtsp://localhost:8554/cam")
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera stream")
            return

        self.get_logger().info("Camera stream opened successfully")
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10fps

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        self.get_logger().info("Published frame")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    with rclpy.init(args=args):
        node = CameraNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()


if __name__ == "__main__":
    main()
