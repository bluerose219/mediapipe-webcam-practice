"""학습한 커스텀 제스처 모델을 웹용 JSON으로 내보내기

실행: python export_web_model.py
결과: web/gesture_model.json  (+ web/hand_landmarker.task 복사)

train_gestures.py로 다시 학습했으면 이 스크립트도 다시 실행해야 웹에 반영됩니다.
"""
import json
import os
import shutil

import joblib

from gesture_common import BASE_DIR, HAND_MODEL_PATH, MODEL_OUT_PATH

WEB_DIR = os.path.join(BASE_DIR, "web")


def main():
    model = joblib.load(MODEL_OUT_PATH)
    scaler = model.named_steps["standardscaler"]
    mlp = model.named_steps["mlpclassifier"]

    data = {
        "classes": [str(c) for c in mlp.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,          # 은닉층: relu
        "out_activation": mlp.out_activation_,  # 출력층: softmax
        "weights": [w.tolist() for w in mlp.coefs_],      # [입력 수][출력 수]
        "biases": [b.tolist() for b in mlp.intercepts_],
    }

    os.makedirs(WEB_DIR, exist_ok=True)
    out_path = os.path.join(WEB_DIR, "gesture_model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    shutil.copyfile(HAND_MODEL_PATH, os.path.join(WEB_DIR, "hand_landmarker.task"))

    print(f"저장: {out_path}")
    print("클래스:", ", ".join(data["classes"]))
    print("층 구조:", " → ".join(str(len(w)) for w in data["weights"]), "→", len(data["classes"]))


if __name__ == "__main__":
    main()
