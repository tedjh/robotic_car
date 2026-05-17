# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import tty
import termios

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

COMMANDS = {
    "w": "W",  # forward
    "r": "R",  # backward
    "a": "A",  # left
    "d": "D",  # right
    "s": "S",  # stop
    "q": None,  # quit
}

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        self.publisher_ = self.create_publisher(String, 'velocity', 10)
        self.get_logger().info('Teleop node started, use W/A/S/D for movement, Ctrl+C to quit')
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.i = 0

    #def timer_callback(self):
    #    msg = String()
    #    msg.data = 'Hello World: %d' % self.i
    #    self.publisher_.publish(msg)
    #    self.get_logger().info('Publishing: "%s"' % msg.data)
    #    self.i += 1
    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            while rclpy.ok():
                key = get_key(settings)
                
                if key == '\x03':  # Ctrl+C
                    break
                cmd = COMMANDS.get(key)
                if cmd is not None:
                    msg = String()
                    msg.data = cmd
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published command: {repr(cmd)}')
        finally:
            # Restore terminal settings on exit
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    with rclpy.init(args=args):
        controller = Controller()
        try:
            controller.run()
        finally:
            controller.destroy_node()


if __name__ == '__main__':
    main()
