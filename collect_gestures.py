"""커스텀 제스처 데이터 수집 - 웹캠으로 손 랜드마크를 모아 data/gestures.csv에 저장

실행: python collect_gestures.py none heart ok rock
      (인자로 준 이름들이 클래스가 됨. 순서대로 숫자키 1, 2, 3 ...에 연결)

조작:
  1~9   : 수집할 제스처 선택
  SPACE : 녹화 시작 / 정지 (녹화 중에는 손이 보이는 프레임마다 저장)
  q/ESC : 종료

이미 있는 data/gestures.csv에 이어서 저장하므로 여러 번 나눠 모아도 됩니다.
"""
import csv
import os
import sys
import time
from collections import Counter

import cv2
import mediapipe as mp

from gesture_common import DATA_PATH, create_hand_landmarker, draw_hand, landmarks_to_features


def load_counts():
    if not os.path.exists(DATA_PATH):
        return Counter()
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return Counter(row[0] for row in csv.reader(f) if row)


def main():
    labels = sys.argv[1:]
    if not labels or len(labels) > 9:
        print("사용법: python collect_gestures.py <제스처1> <제스처2> ... (최대 9개)")
        print("예시:   python collect_gestures.py none heart ok rock")
        sys.exit(1)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    counts = load_counts()
    current = 0
    recording = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    start = time.monotonic()
    with create_hand_landmarker(num_hands=1) as landmarker, \
            open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, landmarks, (0, 0, 255) if recording else (255, 255, 255))
                if recording:
                    features = landmarks_to_features(landmarks, handedness)
                    writer.writerow([labels[current], *(f"{v:.5f}" for v in features)])
                    counts[labels[current]] += 1

            # 안내 문구
            status = "REC" if recording else "PAUSE"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{status}] {current + 1}: {labels[current]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for i, name in enumerate(labels):
                mark = ">" if i == current else " "
                cv2.putText(frame, f"{mark}{i + 1}. {name}: {counts[name]}", (10, 65 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, "1-9: select  SPACE: rec  q: quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Collect Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                current = key - ord("1")
                recording = False  # 제스처를 바꾸면 녹화는 일단 멈춤

    cap.release()
    cv2.destroyAllWindows()
    print(f"저장 위치: {DATA_PATH}")
    for name in labels:
        print(f"  {name}: {counts[name]}개")


if __name__ == "__main__":
    main()
