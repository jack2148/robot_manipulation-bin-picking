import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class KeyboardContinueNode(Node):
    def __init__(self):
        super().__init__("keyboard_continue_node")

        self.pub = self.create_publisher(Empty, "/manual_continue", 10)

        self.get_logger().info(
            "Press SPACE in this terminal to publish /manual_continue. "
            "Press Ctrl+C to exit."
        )

    def run(self):
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY. Run this node directly in a terminal.")
            return

        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while rclpy.ok():
                key = sys.stdin.read(1)

                if key == " ":
                    self.pub.publish(Empty())
                    self.get_logger().info("Published /manual_continue")

                elif key == "\x03":
                    raise KeyboardInterrupt

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)

    node = KeyboardContinueNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()