import rclpy
from std_msgs.msg import Int32


class GripperInterface:
    """
    /grip_state publish 기반 그리퍼 제어 담당.
    """

    def __init__(
        self,
        node,
        topic: str,
        grip_open: int,
        grip_close: int,
        grip_stop: int,
    ):
        self.node = node
        self.GRIP_OPEN = int(grip_open)
        self.GRIP_CLOSE = int(grip_close)
        self.GRIP_STOP = int(grip_stop)
        self.pub = self.node.create_publisher(Int32, topic, 10)

    def publish(self, state_value: int):
        msg = Int32()
        msg.data = int(state_value)
        self.pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.05)

    def open(self):
        self.node.get_logger().info("[GRIP] OPEN")
        self.publish(self.GRIP_OPEN)

    def close(self):
        self.node.get_logger().info("[GRIP] CLOSE")
        self.publish(self.GRIP_CLOSE)

    def stop(self):
        self.node.get_logger().info("[GRIP] STOP")
        self.publish(self.GRIP_STOP)
