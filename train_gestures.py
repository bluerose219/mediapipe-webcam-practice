"""커스텀 제스처 학습 - data/gestures.csv로 분류 모델을 학습해 models/custom_gesture.joblib에 저장

실행: python train_gestures.py
"""
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_common import DATA_PATH, MODEL_OUT_PATH


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} 가 없습니다. 먼저 collect_gestures.py로 데이터를 모으세요.")
    labels = np.loadtxt(DATA_PATH, delimiter=",", usecols=0, dtype=str, encoding="utf-8")
    features = np.loadtxt(DATA_PATH, delimiter=",", usecols=range(1, 64), dtype=np.float32, encoding="utf-8")
    return features, labels


def main():
    X, y = load_data()
    counts = Counter(y)
    print("클래스별 데이터 수:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    if len(counts) < 2:
        raise ValueError("제스처가 2개 이상 있어야 학습할 수 있습니다.")

    # 80%로 학습, 20%로 성능 확인 (클래스 비율 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                      early_stopping=True, random_state=42),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n테스트 정확도: {np.mean(y_pred == y_test):.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    classes = model.classes_
    print("혼동 행렬 (행: 정답, 열: 예측)")
    print("        " + " ".join(f"{c[:6]:>6}" for c in classes))
    for name, row in zip(classes, confusion_matrix(y_test, y_pred, labels=classes)):
        print(f"{name[:6]:>6}  " + " ".join(f"{v:>6}" for v in row))

    # 테스트로 확인했으니 전체 데이터로 다시 학습해서 저장
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUT_PATH)
    print(f"\n모델 저장: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
