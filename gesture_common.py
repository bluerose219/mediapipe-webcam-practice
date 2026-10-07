"""커스텀 제스처 수집/학습/추론에서 같이 쓰는 함수 모음"""
import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_PATH = os.path.join(BASE_DIR, "data", "gestures.csv")
MODEL_OUT_PATH = os.path.join(BASE_DIR, "models", "custom_gesture.joblib")

# 손가락 뼈대 연결 (21개 랜드마크 인덱스)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),                                 # 손바닥
]


def create_hand_landmarker(num_hands=1):
    # 한글 경로에서 파일 열기 문제를 피하려고 모델을 바이트로 읽어서 전달
    with open(HAND_MODEL_PATH, "rb") as f:
        model_data = f.read()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_data),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(landmarks, handedness):
    """손 랜드마크 21개 → 학습용 특징 벡터 63개 (x, y, z)

    - 손목(0번)을 원점으로 옮겨서 화면 속 손 위치와 무관하게 만듦
    - 가장 먼 점까지 거리로 나눠서 손 크기(카메라와의 거리)와 무관하게 만듦
    - 왼손은 x를 뒤집어서 오른손과 같은 모양으로 맞춤 → 한 손으로만 모아도 양손 인식
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_hand(frame, landmarks, color=(255, 255, 255)):
    """손 뼈대를 그리고, 손 영역 왼쪽 위 좌표를 돌려줌 (글자 표시용)"""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return min(p[0] for p in pts), min(p[1] for p in pts)
