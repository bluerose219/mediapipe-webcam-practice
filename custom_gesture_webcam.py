"""커스텀 제스처 추론 - 학습한 models/custom_gesture.joblib로 웹캠에서 제스처 인식

실행: python custom_gesture_webcam.py   (종료: q 또는 ESC)
"""
import time

import cv2
import joblib
import mediapipe as mp

from gesture_common import MODEL_OUT_PATH, create_hand_landmarker, draw_hand, landmarks_to_features

THRESHOLD = 0.7  # 확률이 이보다 낮으면 "?"로 표시


def main():
    model = joblib.load(MODEL_OUT_PATH)
    print("인식할 제스처:", ", ".join(model.classes_))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with create_hand_landmarker(num_hands=2) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand = handedness[0].category_name
                features = landmarks_to_features(landmarks, hand)
                proba = model.predict_proba([features])[0]
                best = proba.argmax()
                name = model.classes_[best] if proba[best] >= THRESHOLD else "?"

                x0, y0 = draw_hand(frame, landmarks)
                cv2.putText(frame, f"{hand}: {name} ({proba[best]:.2f})", (x0, max(y0 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
