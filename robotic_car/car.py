import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


class Car(Node):
    def __init__(self):
        super().__init__("car")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            String, "velocity", self.listener_callback, qos
        )
        self.ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)
        self.current_state = None  # Track the current state of the car

    def listener_callback(self, msg):
        if msg.data != self.current_state:
            self.get_logger().info('I heard: "%s"' % msg.data)
            self.current_state = msg.data
        self.ser.write(msg.data.encode())


def main(args=None):
    with rclpy.init(args=args):
        car = Car()
        try:
            rclpy.spin(car)
        except (KeyboardInterrupt, ExternalShutdownException):
            car.ser.close()  # type: ignore


if __name__ == "__main__":
    main()
