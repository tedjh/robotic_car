import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String


class Car(Node):

    def __init__(self):
        super().__init__('car')
        self.subscription = self.create_subscription(
            String,
            'velocity',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    try:
        with rclpy.init(args=args):
            car = Car()

            rclpy.spin(car)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()