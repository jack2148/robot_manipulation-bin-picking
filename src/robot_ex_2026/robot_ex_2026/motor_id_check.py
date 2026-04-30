from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = "/dev/ttyUSB0"   
BAUDRATE   = 57600             
PROTOCOL_VERSION = 2.0      

ADDR_MODEL_NUMBER = 0

def main():
    port = PortHandler(DEVICENAME)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        print("포트 open 실패:", DEVICENAME)
        return
    if not port.setBaudRate(BAUDRATE):
        print("baudrate 설정 실패:", BAUDRATE)
        return

    found = []
    for dxl_id in range(0, 253):  
        model, dxl_comm_result, dxl_error = packet.read2ByteTxRx(
            port, dxl_id, ADDR_MODEL_NUMBER
        )
        if dxl_comm_result == 0 and dxl_error == 0:
            found.append((dxl_id, model))

    port.closePort()

    if not found:
        print("발견된 다이나믹셀이 없음. 포트/baud/전원/Protocol 확인")
    else:
        print("현재 ID:")
        for dxl_id, model in found:
            print(f"  ID={dxl_id:3d}  ModelNumber=0x{model:04X} ({model})")

if __name__ == "__main__":
    main()
