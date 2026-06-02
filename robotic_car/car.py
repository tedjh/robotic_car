import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import serial

from std_msgs.msg import String


class Car(Node):

    def __init__(self):
        super().__init__('car')
        self.subscription = self.create_subscription(
            String,
            'velocity',
            self.listener_callback,
            10)
        self.subscription
        self.ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    
    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        
        self.ser.write(msg.data.encode())

def main(args=None):
    car = Car()
    try:
        with rclpy.init(args=args):
            rclpy.spin(car)
    except (KeyboardInterrupt, ExternalShutdownException):
        car.ser.close()


if __name__ == '__main__':
    main()