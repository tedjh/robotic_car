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
import select
import sys
import termios
import tty
from collections import deque
from dataclasses import dataclass, field
from time import sleep

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class Action:
    left_speed: int
    left_direction: int
    right_speed: int
    right_direction: int

    def __post_init__(self):
        # Ensure that speed values are within the valid range (0-255)
        self.left_speed = max(0, min(255, self.left_speed))
        self.right_speed = max(0, min(255, self.right_speed))
        self.left_direction = 1 if self.left_direction else 0
        self.right_direction = 1 if self.right_direction else 0

    def to_command_string(self) -> str:
        return f"{self.left_speed},{self.left_direction},{self.right_speed},{self.right_direction}"


@dataclass
class COMMANDS:
    forward: Action = field(default_factory=lambda: Action(150, 1, 150, 1))  # forward
    backward: Action = field(default_factory=lambda: Action(150, 0, 150, 0))  # backward
    left: Action = field(default_factory=lambda: Action(70, 1, 150, 1))  # left
    right: Action = field(default_factory=lambda: Action(150, 1, 70, 1))  # right
    stop: Action = field(default_factory=lambda: Action(0, 1, 0, 1))  # stop

    def get_string_command(self, key: str) -> str | None:
        match key:
            case "w":
                return self.forward.to_command_string()
            case "r":
                return self.backward.to_command_string()
            case "a":
                return self.left.to_command_string()
            case "d":
                return self.right.to_command_string()
            case "s":
                return self.stop.to_command_string()
            case _:
                return None

    def get_action(self, key: str) -> Action | None:
        match key:
            case "w":
                return self.forward
            case "r":
                return self.backward
            case "a":
                return self.left
            case "d":
                return self.right
            case "s":
                return self.stop
            case _:
                return None


def get_key(settings) -> str | None:
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def get_key_no_blocking(settings) -> str | None:
    tty.setraw(sys.stdin.fileno())
    # Check if input is available with a 0.1s timeout
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class Controller(Node):
    def __init__(self):
        super().__init__("controller")
        self.publisher_ = self.create_publisher(String, "velocity", 10)
        self.get_logger().info(
            "Teleop node started, use W/A/S/D for movement, Ctrl+C to quit"
        )

        self.command_wait_time = 0.1  # seconds
        self.action_buffer: deque[Action] = deque([])
        self.current_state: Action = COMMANDS().stop  # Start in stopped state
        self.number_transition_steps: int = 5

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            while rclpy.ok():
                # Retrieve latest key input by user.
                key = get_key_no_blocking(settings)
                # self.get_logger().info("key pressed: %s" % repr(key))

                # Break out if Ctrl+C is pressed.
                if key == "\x03":  # Ctrl+C
                    break

                # If a key was pressed by user, update the action buffer accordingly.
                self.update_action_buffer(key)

                # Get next action to perform, and remove from the buffer.
                cmd = self.next_action()

                self.current_state = cmd  # Update current state to the new command
                # Publish the command to the 'velocity' topic.
                msg = String(data=f"{cmd.to_command_string()}\n")
                self.publisher_.publish(msg)
                self.get_logger().info(f"Published command: {repr(cmd)}")
                sleep(
                    self.command_wait_time
                )  # Small delay to prevent flooding the topic
        finally:
            # Restore terminal settings on exit
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def update_action_buffer(self, key: str | None) -> None:
        # If no new key was pressed, do not update the action buffer.
        if key is None:
            return

        target_action = COMMANDS().get_action(key)

        # No need to update if the command is the same as current state
        if target_action is None or self.current_state == target_action:
            # self.get_logger().info("No update needed.")
            return

        # Clear existing buffer to prioritize new command
        self.action_buffer.clear()

        # Plan the transition from current state to target action, treating left and
        # right motors separately.
        left_buffer = self.plan_motor_transition(
            self.current_state.left_speed,
            self.current_state.left_direction,
            target_action.left_speed,
            target_action.left_direction,
        )
        right_buffer = self.plan_motor_transition(
            self.current_state.right_speed,
            self.current_state.right_direction,
            target_action.right_speed,
            target_action.right_direction,
        )

        # Populate the action buffer with intermediate steps.
        self.action_buffer.extend(
            Action(
                left_speed=left_cmd[0],
                left_direction=left_cmd[1],
                right_speed=right_cmd[0],
                right_direction=right_cmd[1],
            )
            for left_cmd, right_cmd in zip(left_buffer, right_buffer)
        )
        self.get_logger().info(
            f"Updated action buffer with {len(self.action_buffer)} intermediate steps."
        )

    def plan_motor_transition(
        self,
        current_speed: int,
        current_direction: int,
        target_speed: int,
        target_direction: int,
    ) -> list[tuple[int, int]]:
        buffer = []
        # self.get_logger().info(
        #    f"Planning motor transition: current_speed={current_speed}, current_direction={current_direction}, "
        #    f"target_speed={target_speed}, target_direction={target_direction}"
        # )
        # Handle the case that motor direction is remaining the same.
        if current_direction == target_direction:
            increment: int = (
                target_speed - current_speed
            ) // self.number_transition_steps
            buffer.extend(
                [
                    (
                        current_speed + increment * i,
                        current_direction,
                    )
                    for i in range(1, self.number_transition_steps)
                ]
            )

            # Ensure the final target action is included precisely, even if there are
            # rounding issues with the increments
            buffer.append((target_speed, target_direction))

        # Handle the case that motor direction is changing, in which case we need to
        # first decelerate to 0 before accelerating in the opposite direction.
        else:
            increment = (target_speed + current_speed) // self.number_transition_steps
            for i in range(1, self.number_transition_steps):
                raw_new_speed = current_speed - increment * i
                if raw_new_speed >= 0:
                    buffer.append(
                        (
                            raw_new_speed,
                            current_direction,
                        )
                    )
                else:
                    buffer.append(
                        (
                            -raw_new_speed,
                            target_direction,
                        )
                    )
            buffer.append((target_speed, target_direction))
        return buffer

    def next_action(self) -> Action:
        return (
            self.action_buffer.popleft() if self.action_buffer else self.current_state
        )


def main(args=None):
    with rclpy.init(args=args):
        controller = Controller()
        try:
            controller.run()
        finally:
            controller.destroy_node()


if __name__ == "__main__":
    main()
