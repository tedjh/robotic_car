import rclpy
from cv_bridge import CvBridge
from picamera2 import Picamera2  # type: ignore # Only exists on Raspberry Pi.
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10fps

    def timer_callback(self):
        frame = self.picam2.capture_array()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
