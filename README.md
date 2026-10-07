# MediaPipe 웹캠 실습 정리 (2026-10-07)

오늘 수업에서는 Google MediaPipe Tasks의 비전 모델 3가지를 웹캠에 연결해 실시간으로 돌려 봤습니다.

| 순서 | 실습 | 모델 파일 | 검출 결과 |
|---|---|---|---|
| 1 | Hand Landmarker | `hand_landmarker.task` | 손 21개 랜드마크 + 왼손/오른손 |
| 2 | Gesture Recognizer | `gesture_recognizer.task` | 손 랜드마크 + 제스처 이름 |
| 3 | Face Landmarker | `face_landmarker.task` | 얼굴 478개 랜드마크 + blendshape |

이 저장소에는 3번 Face Landmarker 코드(`face_landmarker_webcam.py`)가 들어 있습니다.

---

## 1. 환경 준비

```
pip install -r requirements.txt   # mediapipe, opencv-python
```

- 버전: Python 3.14, mediapipe 1.1.0, opencv-python 5.0.0
- 모델(`.task`) 파일은 각 솔루션 문서 페이지의 **Models** 항목에서 받습니다.
  - Face: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- mediapipe 1.x부터는 예전 `mp.solutions` (`drawing_utils` 등)가 없습니다. 그래서 랜드마크는 OpenCV(`cv2.line`, `cv2.circle`)로 직접 그립니다.

## 2. MediaPipe Tasks 공통 흐름

세 모델 모두 사용 순서가 같습니다. 클래스 이름과 검출 함수 이름만 다릅니다.

```python
import cv2, time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1) 모델 읽기 (한글 경로 문제 때문에 바이트로 전달 → 아래 '문제 해결' 참고)
with open("face_landmarker.task", "rb") as f:
    model_data = f.read()

# 2) 옵션 설정
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=model_data),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
)

# 3) 모델 생성 → 4) 프레임마다 검출
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
start = time.monotonic()
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                         # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # OpenCV는 BGR, MediaPipe는 RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - start) * 1000)        # 밀리초 타임스탬프 (계속 증가해야 함)
        result = landmarker.detect_for_video(mp_image, ts)
        # 5) result를 이용해 그리기 ...
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):        # q 또는 ESC로 종료
            break
cap.release()
cv2.destroyAllWindows()
```

### 모델별 클래스·함수 이름

| 모델 | Options / 클래스 | VIDEO 모드 검출 함수 | 개수 옵션 |
|---|---|---|---|
| Hand Landmarker | `HandLandmarkerOptions` / `HandLandmarker` | `detect_for_video` | `num_hands` |
| Gesture Recognizer | `GestureRecognizerOptions` / `GestureRecognizer` | `recognize_for_video` | `num_hands` |
| Face Landmarker | `FaceLandmarkerOptions` / `FaceLandmarker` | `detect_for_video` | `num_faces` |

### Running mode 3가지

| 모드 | 용도 | 호출 방식 |
|---|---|---|
| `IMAGE` | 사진 한 장 | `detect(image)` |
| `VIDEO` | 동영상·웹캠 프레임 (동기) | `detect_for_video(image, timestamp_ms)` |
| `LIVE_STREAM` | 실시간 스트림 (비동기) | `detect_async(...)` + `result_callback` |

오늘 실습에서는 코드가 단순한 `VIDEO` 모드를 썼습니다. 타임스탬프가 이전 프레임보다 커야 하므로 `time.monotonic()`으로 계산합니다.

### 좌표 변환

결과 랜드마크의 `x`, `y`는 0~1로 **정규화된 값**입니다. 화면에 그리려면 이미지 크기를 곱해 픽셀 좌표로 바꿉니다.

```python
h, w = frame.shape[:2]
pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
```

---

## 3. 실습 1 — Hand Landmarker (손 랜드마크)

- 손 하나당 **21개 랜드마크**: 0번은 손목, 4·8·12·16·20번은 각 손가락 끝
- `result.hand_landmarks`: 손별 랜드마크 리스트
- `result.handedness`: 왼손/오른손 (`"Left"` / `"Right"`)과 확률
- 주요 옵션: `num_hands`, `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence` (기본 0.5)

손가락 뼈대 연결 인덱스:

```python
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),                                 # 손바닥
]
```

> 거울 모드(`cv2.flip`)로 뒤집은 영상을 넣으면 화면에 보이는 손 기준으로 Left/Right가 나옵니다.

## 4. 실습 2 — Gesture Recognizer (손 제스처 인식)

- Hand Landmarker의 결과에 **제스처 분류**가 추가된 모델
- 인식 가능한 기본 제스처 7개 (+ `None`):
  `Closed_Fist` ✊, `Open_Palm` ✋, `Pointing_Up` ☝️, `Thumb_Down` 👎, `Thumb_Up` 👍, `Victory` ✌️, `ILoveYou` 🤟
- `result.gestures[i][0].category_name`, `.score`로 i번째 손의 제스처와 확률을 읽음
- 검출 함수 이름이 `detect_for_video`가 아니라 **`recognize_for_video`** 인 점에 주의

## 5. 실습 3 — Face Landmarker (얼굴 랜드마크) ← 이 저장소

```
python face_landmarker_webcam.py
```

- 얼굴 하나당 **478개 랜드마크** (468개 얼굴 메시 + 홍채 10개)
- `output_face_blendshapes=True`로 켜면 표정 계수 **52개**(첫 번째는 `_neutral`)를 0~1 점수로 받음
  (예: `eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft`, `browInnerUp`)
- `output_facial_transformation_matrixes=True`로 켜면 머리 회전·위치 행렬도 받을 수 있음 (고개 방향 추정용)
- 연결선은 `vision.FaceLandmarksConnections`에 그룹별로 정의되어 있음 (모두 `FACE_LANDMARKS_` 접두사):
  `TESSELATION`(메시), `CONTOURS`, `FACE_OVAL`, `LIPS`, `NOSE`, `LEFT_EYE`, `RIGHT_EYE`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`, `LEFT_IRIS`, `RIGHT_IRIS`

이 코드의 화면 구성:
- 메시(회색), 얼굴 윤곽, 입술, 눈·눈썹(왼쪽 초록 / 오른쪽 빨강), 홍채(파랑)
- 첫 번째 얼굴의 blendshape 상위 5개 점수
- FPS, 검출된 얼굴 수

---

## 6. 문제 해결 (오늘 겪은 것)

### "웹캠이 안 열려요" → 실제로는 모델 파일 경로 문제

```
RuntimeError: Unable to open file at C:\Users\...\바탕 화면\...\face_landmarker.task, errno=-1
```

- 원인: 경로에 **한글 같은 비ASCII 문자**가 있으면 MediaPipe 내부(C 라이브러리)가 `model_asset_path`의 파일을 열지 못함. 모델 생성 단계에서 에러가 나서 웹캠 창이 뜨기 전에 프로그램이 꺼지므로, 웹캠 문제처럼 보임.
- 해결: 파이썬으로 파일을 직접 읽어 `model_asset_buffer`로 전달

```python
with open(MODEL_PATH, "rb") as f:
    model_data = f.read()
base_options = python.BaseOptions(model_asset_buffer=model_data)
```

### 진짜 웹캠이 안 열릴 때

- `cv2.VideoCapture(0, ...)`의 번호를 `1`로 바꿔 보기 (카메라가 여러 개인 노트북)
- Zoom, Teams, 브라우저 등 카메라를 쓰고 있는 프로그램 종료
- Windows 설정 → 개인 정보 → 카메라에서 데스크톱 앱 접근 허용 확인

### 무시해도 되는 로그

실행할 때 나오는 `WARNING: Logging before InitGoogle()`, `Created TensorFlow Lite XNNPACK delegate` 같은 메시지는 정상 동작 중에 나오는 로그입니다.

---

## 참고 문서

- Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Gesture Recognizer: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
- 모델 라이선스: Apache 2.0 (Google)
