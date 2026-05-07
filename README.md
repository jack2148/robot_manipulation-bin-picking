# RGB-D 카메라 기반 실시간 3D 물체 탐지 및 자세 추정

Intel RealSense D455 카메라로 산업용 부품 3종(cylinder / hole / cross)을 실시간 인식하고, 각 물체의 **3D 좌표(X, Y, Z)**와 **방향각(angle)**을 추출하는 시스템입니다.

---

## 개요

공장 자동화나 로봇 pick-and-place 작업에서 물체의 위치와 방향을 정확히 알아야 그리퍼가 제대로 접근할 수 있습니다. 이 프로젝트는 RGB-D 카메라의 깊이(depth) 정보를 활용해 2D 이미지 인식을 넘어 실제 공간상의 3D 위치를 뽑아내는 것을 목표로 만들었습니다.

YOLOv8 instance segmentation으로 마스크를 생성하고, 마스크 영역의 depth 중앙값과 카메라 내부 파라미터를 이용해 픽셀 좌표를 3D 좌표로 변환합니다. 방향각은 마스크의 minAreaRect로 주축 방향을 계산합니다.

---

## 파이프라인

```
데이터 수집          라벨링            학습             추론
(data_collector) → (Roboflow) → (train_yolo.py) → (detect_3d_pose.py)
RealSense D455      polygon seg      YOLOv8n-seg      실시간 3D 위치 + 각도
```

### 단계별 설명

**1. 데이터 수집**
`data_collector.py`를 실행하면 RealSense 컬러 스트림이 뜨고, `r` 키로 0.5초 간격 자동 촬영, `s` 키로 수동 1장 저장이 됩니다. 화이트 밸런스를 고정해 조명 변화에 일관된 색감을 유지합니다.

**2. 라벨링**
수집한 이미지를 Roboflow에 업로드해 polygon segmentation 형식으로 라벨링했습니다. 클래스별로 분리된 데이터셋을 export합니다.

**3. 데이터셋 병합 및 학습**
`train_yolo.py`는 cylinder / hole / cross 3개의 개별 데이터셋을 class ID를 remapping해 하나로 합친 뒤 YOLOv8n-seg를 학습합니다. 파일명 충돌 방지를 위해 클래스명을 prefix로 붙입니다.

```python
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
)
```

**4. 실시간 추론**
`detect_3d_pose.py`가 컬러/뎁스 프레임을 정합(align)한 뒤 YOLO로 마스크를 생성하고, 아래 순서로 3D 자세를 계산합니다.

- 마스크 무게중심 → 픽셀 좌표 (cx, cy)
- 마스크 영역 depth 중앙값 → depth_m (m)
- 카메라 내부 파라미터로 역투영 → (X, Y, Z)
- `minAreaRect` 긴 변 방향 → angle (deg)

---

## 데이터셋

| 클래스   | train | valid |
|----------|------:|------:|
| cylinder |   112 |    21 |
| hole     |   137 |    25 |
| cross    |   141 |    26 |

---

## 학습 결과

### 학습 곡선

![Training Curves](training_curves.png)

20 epoch 이전에 mAP50이 0.9 이상으로 빠르게 수렴하고, 이후 100 epoch까지 안정적으로 유지됩니다. train/val loss가 함께 감소하며 과적합 없이 수렴했습니다.

### 클래스별 Validation 성능

![Best Metrics per Class](training_best_metrics.png)

![Summary Table](training_summary_table.png)

| Class    | mAP50 (Box) | mAP50 (Mask) | Precision | Recall |
|----------|:-----------:|:------------:|:---------:|:------:|
| cylinder |   0.9950    |    0.9950    |   0.9839  | 1.0000 |
| hole     |   0.8473    |    0.8394    |   0.9458  | 0.8571 |
| cross    |   0.9527    |    0.9527    |   0.9062  | 0.9666 |
| **mean** | **0.9317**  |  **0.9290**  | **0.9453**|**0.9412**|

cylinder와 cross는 형태가 뚜렷해 mAP50 0.95 이상을 달성했습니다. hole은 촬영 각도에 따라 형태 변화가 커서 상대적으로 낮지만 Precision은 0.946으로 오검출은 적습니다.

---

## 실행 방법

```bash
# 1. 패키지 설치
pip install ultralytics pyrealsense2 opencv-python numpy

# 2. 데이터 수집
python data_collector.py

# 3. 학습 (Roboflow 데이터셋을 cylinder/, hole/, cross/ 에 준비 후)
python train_yolo.py

# 4. 실시간 추론 (RealSense D455 연결 필요)
python detect_3d_pose.py
```

종료는 ESC 키입니다.

---

## 파일 구조

```
rgbd_camera/
├── data_collector.py      # RealSense 이미지 수집 도구
├── train_yolo.py          # 데이터셋 병합 + YOLOv8 학습
├── detect_3d_pose.py      # 실시간 3D 탐지 메인 스크립트
├── analyze_results.py     # 학습 결과 시각화 (곡선, 표, 막대그래프)
├── cylinder/              # cylinder 클래스 데이터셋 (Roboflow export)
├── hole/                  # hole 클래스 데이터셋
├── cross/                 # cross 클래스 데이터셋
├── object/                # 병합 데이터셋
├── runs/                  # YOLO 학습 결과 (best.pt 포함)
├── training_curves.png
├── training_best_metrics.png
└── training_summary_table.png
```

---

## 모드 전환 (object ↔ insert)

`detect_3d_pose.py`는 두 가지 모드를 지원합니다.

| 모드 | 탐지 클래스 | ROS 퍼블리시 토픽 | 모델 경로 |
|------|------------|-----------------|-----------|
| `object` | cross, cylinder, hole | `/object_poses` | `runs/segment/train/weights/best.pt` |
| `insert` | cross_insert, cylinder_insert, hole_insert | `/insert_poses` | `runs/segment/insert_seg/weights/best.pt` |

기본 모드는 `object`이며, 실행 중 ROS2 토픽으로 전환합니다.

```bash
# object 모드로 전환
ros2 topic pub --once /detect_mode std_msgs/msg/String "data: 'object'"

# insert 모드로 전환
ros2 topic pub --once /detect_mode std_msgs/msg/String "data: 'insert'"
```

현재 모드는 화면 좌상단에 `MODE: object` / `MODE: insert` 로 표시됩니다.

---

## 개발 환경

- Python 3.10
- Intel RealSense D455
- Ultralytics YOLOv8n-seg
- pyrealsense2, OpenCV, NumPy, Matplotlib
- CUDA GPU (NVIDIA GeForce RTX 4060 Laptop)
