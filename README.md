# Robot Manipulation - Bin Picking

RealSense D455 카메라와 YOLOv8 segmentation 모델을 이용해 bin picking 환경에서 물체(cross, cylinder, hole)의 **3D 위치(X, Y, Z)와 orientation**을 인식하고 ROS2 토픽으로 발행하는 프로젝트입니다.

---

## 환경

- Python 3.10+
- ROS2 (Humble 이상)
- Intel RealSense D455

### 의존성 설치

```bash
pip install ultralytics pyrealsense2 opencv-python numpy
```

---

## 실행

ROS2 환경에서 RealSense D455를 연결한 후 실행합니다.

```bash
python pose_publisher.py
```

- 토픽: `/object_poses` (`std_msgs/String`, JSON 형식)
- 발행 주기: 10Hz

### 토픽 메시지 형식

```json
{
  "objects": [
    {
      "class": "cylinder",
      "confidence": 0.923,
      "position": { "x": 0.012, "y": -0.045, "z": 0.631 },
      "orientation": {
        "axis_x": [0.998, 0.023, -0.012],
        "axis_y": [-0.021, 0.994, 0.108],
        "axis_z": [0.015, -0.107, 0.994]
      }
    }
  ]
}
```

### 토픽 확인

```bash
ros2 topic echo /object_poses
```

---

## 인식 클래스

| ID | 클래스 | 색상 |
|----|--------|------|
| 0 | cross | 초록 |
| 1 | cylinder | 주황 |
| 2 | hole | 보라 |
