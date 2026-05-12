# vision

RealSense D455 카메라로 YOLO 세그멘테이션 모델을 사용해 물체 위치·자세를 추정하고 ROS2 토픽으로 발행하는 패키지.

---

## 모델

| 파일 | 설명 |
|------|------|
| `weights/best.pt` | object 전용 모델 (cylinder, hole, cross) |
| `weights/insert_best.pt` | insert 전용 모델 (cross_insert, cylinder_insert, hole_insert) |
| `weights/ob_in_best.pt` | **통합 모델** (object + insert 6클래스) |

---

## 노드

### `pose_publisher_ob_in` (통합 - 권장)

object + insert 6개 클래스를 단일 모델로 동시 검출.
모드 전환 없이 항상 전체 클래스를 인식.

```bash
ros2 launch vision pose_publisher_ob_in.launch.py
```

**발행 토픽 3개:**

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/ob_in_poses` | `std_msgs/String` (JSON) | object + insert 전체 검출 결과 |
| `/object_poses` | `std_msgs/String` (JSON) | object 클래스만 (cylinder, hole, cross) |
| `/insert_poses` | `std_msgs/String` (JSON) | insert 클래스만 (cross_insert, cylinder_insert, hole_insert) |

**JSON 메시지 구조:**
```json
{
  "objects": [
    {
      "class": "cylinder",
      "confidence": 0.923,
      "position": { "x": 0.012, "y": -0.034, "z": 0.541 },
      "orientation": {
        "axis_x": [0.99, 0.01, 0.0],
        "axis_y": [0.01, 0.99, 0.0],
        "axis_z": [0.0, 0.0, 1.0]
      }
    }
  ]
}
```

**클래스 목록:**

| ID | 이름 | 색상 |
|----|------|------|
| 0 | cylinder | 주황 |
| 1 | hole | 보라 |
| 2 | cross | 초록 |
| 3 | cross_insert | 진초록 |
| 4 | cylinder_insert | 파랑 |
| 5 | hole_insert | 진보라 |

---

### `pose_publisher` (구 버전 - object / insert 분리)

`/detect_mode` 토픽으로 모드 전환 필요.

```bash
ros2 launch vision pose_publisher.launch.py
# 모드 전환 예시
ros2 topic pub /detect_mode std_msgs/String "data: 'insert'"
```

---

## 빌드

```bash
cd /home/chan/robot_manipulation-bin-picking
colcon build --packages-select vision
source install/setup.bash
```
