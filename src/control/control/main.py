import rclpy

try:
    from .peg_in_hole_controller import PegInHoleController
except ImportError:  # direct script/debug execution support
    from peg_in_hole_controller import PegInHoleController


def main():
    rclpy.init()
    node = PegInHoleController()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Exit")


if __name__ == "__main__":
    main()
