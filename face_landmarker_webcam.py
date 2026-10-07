"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크 검출

실행: python face_landmarker_webcam.py
종료: q 또는 ESC
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
C = vision.FaceLandmarksConnections

# 그릴 연결선 그룹과 색상 (BGR)
CONNECTION_GROUPS = [
    (C.FACE_LANDMARKS_TESSELATION, (80, 80, 80), 1),
    (C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224), 1),
    (C.FACE_LANDMARKS_LIPS, (180, 180, 255), 1),
    (C.FACE_LANDMARKS_LEFT_EYE, (48, 255, 48), 1),
    (C.FACE_LANDMARKS_LEFT_EYEBROW, (48, 255, 48), 1),
    (C.FACE_LANDMARKS_RIGHT_EYE, (48, 48, 255), 1),
    (C.FACE_LANDMARKS_RIGHT_EYEBROW, (48, 48, 255), 1),
    (C.FACE_LANDMARKS_LEFT_IRIS, (255, 200, 0), 2),
    (C.FACE_LANDMARKS_RIGHT_IRIS, (255, 200, 0), 2),
]


def draw_landmarks(image, result):
    h, w = image.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for connections, color, thickness in CONNECTION_GROUPS:
            for conn in connections:
                if conn.start < len(pts) and conn.end < len(pts):
                    cv2.line(image, pts[conn.start], pts[conn.end], color, thickness, cv2.LINE_AA)


def draw_blendshapes(image, result, top_k=5):
    """첫 번째 얼굴의 상위 blendshape 점수 표시 (예: eyeBlinkLeft, jawOpen)"""
    if not result.face_blendshapes:
        return
    top = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)[:top_k]
    for i, cat in enumerate(top):
        y = 60 + i * 22
        cv2.putText(image, f"{cat.category_name}: {cat.score:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서, 직접 읽어 바이트로 전달
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_data),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_landmarks(frame, result)
            draw_blendshapes(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(result.face_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
