"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식
실행: python gesture_webcam.py   (종료: q 또는 ESC)

인식 가능한 제스처: None, Closed_Fist, Open_Palm, Pointing_Up,
                   Thumb_Down, Thumb_Up, Victory, ILoveYou
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")

# 손가락 뼈대 연결 (21개 랜드마크 인덱스)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),                                 # 손바닥
]


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        hand = result.handedness[i][0].category_name  # "Left" / "Right"
        gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        text = f"{hand}: {gesture.category_name} ({gesture.score:.2f})" if gesture else hand

        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        cv2.putText(frame, text, (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    # 한글 경로에서 파일 열기 문제를 피하려고 모델을 바이트로 읽어서 전달
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_data),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_result(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
