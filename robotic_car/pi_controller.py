import rclpy
import torch

from robotic_car.controller_utils import COMMANDS, BaseController


class PiController(BaseController):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("Loading model")

        self.get_logger().info("Use Ctrl+C to quit")

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


def main(args=None):
    with rclpy.init(args=args):
        controller = PiController()
        try:
            controller.run()
        finally:
            controller.destroy_node()


if __name__ == "__main__":
    main()
