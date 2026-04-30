from dynamixel_sdk import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


# ros2 topic pub --once /grip_state std_msgs/msg/Int32 "{data: 1} #열기"
# ros2 topic pub --once /grip_state std_msgs/msg/Int32 "{data: 0} #닫기"
# id 먼저 찾고 DXL_ID 수정 후 사용하세요!
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
DXL_ID = 1
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
# 허용 전류선 확인하고 전류값을 조절해 주세요. xc330 과한 전류는 모터 손상 및 과열의 원인이 됩니다.
CURRENT = 70  

def to_u16(v):
    return v & 0xFFFF

def unpack_result(ret):
    if isinstance(ret, tuple):
        if len(ret) == 2:
            comm, err = ret
            return comm, err
        if len(ret) == 3:
            _, comm, err = ret
            return comm, err
    return -1, 0

class GripCurrentNode(Node):
    def __init__(self):
        super().__init__("grip_current_node")

        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            self.get_logger().error(f"openPort 실패: {DEVICENAME}")
            raise RuntimeError("openPort failed")

        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error(f"setBaudRate 실패: {BAUDRATE}")
            self.portHandler.closePort()
            raise RuntimeError("setBaudRate failed")

        if not self.w1(ADDR_TORQUE_ENABLE, 0):
            raise RuntimeError("Torque OFF failed")
        if not self.w1(ADDR_OPERATING_MODE, 0):  
            raise RuntimeError("Set Operating Mode failed")
        if not self.w1(ADDR_TORQUE_ENABLE, 1):
            raise RuntimeError("Torque ON failed")

        self.last_state = None
        self.last_sent_current = None

        self.sub = self.create_subscription(Int32, "grip_state", self.cb_state, 10)
        self.get_logger().info("0 or 1")

        self.send_current(0)

    def w1(self, addr, val):
        comm, err = unpack_result(self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, addr, val))
        if comm != COMM_SUCCESS:
            self.get_logger().error(f"COMM ERR: {self.packetHandler.getTxRxResult(comm)}")
            return False
        if err != 0:
            self.get_logger().error(f"DXL ERR: {self.packetHandler.getRxPacketError(err)}")
            return False
        return True

    def w2(self, addr, val):
        comm, err = unpack_result(self.packetHandler.write2ByteTxRx(self.portHandler, DXL_ID, addr, to_u16(val)))
        if comm != COMM_SUCCESS:
            self.get_logger().error(f"COMM ERR: {self.packetHandler.getTxRxResult(comm)}")
            return False
        if err != 0:
            self.get_logger().error(f"DXL ERR: {self.packetHandler.getRxPacketError(err)}")
            return False
        return True

    def send_current(self, cur):
        if self.last_sent_current == cur:
            return
        ok = self.w2(ADDR_GOAL_CURRENT, cur)
        if ok:
            self.last_sent_current = cur

    def cb_state(self, msg: Int32):
        state = int(msg.data)

        if state == self.last_state:
            return
        self.last_state = state

        if state == 1:
            self.send_current(-CURRENT)
            self.get_logger().info("grip_on")
        elif state == 0:
            self.send_current(+CURRENT)
            self.get_logger().info("grip_off")
        else:
            self.send_current(0)
            self.get_logger().info(f"grip_state={state} -> STOP")

    def destroy_node(self):
        try:
            self.send_current(0)
            self.w1(ADDR_TORQUE_ENABLE, 0)
            self.portHandler.closePort()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = GripCurrentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()