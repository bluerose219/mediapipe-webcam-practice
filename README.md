# MediaPipe 웹캠 실습 정리 (2026-10-07)

오늘 수업에서는 Google MediaPipe Tasks의 비전 모델 3가지를 웹캠에 연결해 실시간으로 돌려 봤습니다.

| 순서 | 실습 | 실행 코드 | 모델 파일 | 검출 결과 |
|---|---|---|---|---|
| 1 | Hand Landmarker | `hand_webcam.py` | `hand_landmarker.task` | 손 21개 랜드마크 + 왼손/오른손 |
| 2 | Gesture Recognizer | `gesture_webcam.py` | `gesture_recognizer.task` | 손 랜드마크 + 제스처 이름 |
| 3 | Face Landmarker | `face_landmarker_webcam.py` | `face_landmarker.task` | 얼굴 478개 랜드마크 + blendshape |

## 파일 구성

```
├── hand_webcam.py              # 실습 1: 손 랜드마크
├── hand_landmarker.task        #   └ 모델 (7.8MB)
├── gesture_webcam.py           # 실습 2: 손 제스처 인식
├── gesture_recognizer.task     #   └ 모델 (8.4MB)
├── face_landmarker_webcam.py   # 실습 3: 얼굴 랜드마크
├── face_landmarker.task        #   └ 모델 (3.7MB)
├── requirements.txt
└── README.md
```

## 빠른 실행

```
pip install -r requirements.txt
python hand_webcam.py              # 손 랜드마크
python gesture_webcam.py           # 손 제스처
python face_landmarker_webcam.py   # 얼굴 랜드마크
```

모든 프로그램은 `q` 또는 `ESC`로 종료합니다. 모델 파일이 같은 폴더에 있으므로 따로 받을 필요가 없습니다.

---

## 1. 환경 준비

```
pip install -r requirements.txt   # mediapipe, opencv-python
```

- 버전: Python 3.14, mediapipe 1.1.0, opencv-python 5.0.0
- 모델(`.task`) 파일은 각 솔루션 문서 페이지의 **Models** 항목에서 받습니다.
  - Hand: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  - Gesture: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
  - Face: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- 오늘은 Claude Code의 **Claude in Chrome**(크롬 확장)으로 문서 페이지를 열어 모델 파일을 내려받았습니다.
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

```
python hand_webcam.py
```

- 손 하나당 **21개 랜드마크**: 0번은 손목, 4·8·12·16·20번은 각 손가락 끝
- `result.hand_landmarks`: 손별 랜드마크 리스트
- `result.handedness`: 왼손/오른손 (`"Left"` / `"Right"`)과 확률
- 주요 옵션: `num_hands`, `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence` (기본 0.5)
- 화면: 손가락 뼈대(흰 선), 관절(빨간 점), 손 위에 `Left 0.98`처럼 왼손/오른손과 확률, 왼쪽 위에 FPS

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

```
python gesture_webcam.py
```

- Hand Landmarker의 결과에 **제스처 분류**가 추가된 모델
- 인식 가능한 기본 제스처 7개 (+ `None`):
  `Closed_Fist` ✊, `Open_Palm` ✋, `Pointing_Up` ☝️, `Thumb_Down` 👎, `Thumb_Up` 👍, `Victory` ✌️, `ILoveYou` 🤟
- `result.gestures[i][0].category_name`, `.score`로 i번째 손의 제스처와 확률을 읽음
- 검출 함수 이름이 `detect_for_video`가 아니라 **`recognize_for_video`** 인 점에 주의
- 화면: 손 뼈대 위에 `Right: Thumb_Up (0.87)`처럼 손 방향, 제스처, 확률 표시

```python
hand = result.handedness[i][0].category_name
gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
text = f"{hand}: {gesture.category_name} ({gesture.score:.2f})" if gesture else hand
```

## 5. 실습 3 — Face Landmarker (얼굴 랜드마크)

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

## 7. 사용한 프롬프트 (Claude Code)

오늘 이 저장소의 코드는 Claude Code에 아래 프롬프트를 입력해서 만들었습니다. 입력한 그대로 옮겼고, 오타도 고치지 않았습니다. 단, URL 뒤에 붙어 있던 Google 추적용 파라미터(`?_gl=...`)는 지웠습니다.

### 세션 1 — Hand Landmarker, Gesture Recognizer (`미디어` 폴더)

| 시각 | 프롬프트 | 결과 |
|---|---|---|
| 15:23 | `https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker 해당하는 라이브러리 받고 파이썬 실행 코드 webcam 기반으로 짜줘 클로인크롬써서 다운받아` | mediapipe 설치, Chrome으로 `hand_landmarker.task` 다운로드, `hand_webcam.py` 작성 |
| 15:30 | `https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer 여기서 또 해줘` | `gesture_recognizer.task` 다운로드, `gesture_webcam.py` 작성 |

### 세션 2 — Face Landmarker, GitHub, README (`미디어/3` 폴더)

| 시각 | 프롬프트 | 결과 |
|---|---|---|
| 15:33 | `https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker 해당하는 라이브러리 받고 파이썬 실행 코드 webcam 기반으로 짜줘 클로인크롬써서 다운받아` | Chrome으로 `face_landmarker.task` 다운로드, `face_landmarker_webcam.py` 작성 |
| 15:37 | `웹캡이 안 열려` | 원인이 한글 경로임을 찾아 `model_asset_buffer` 방식으로 수정 |
| 15:39 | `이제 잘 된다` | 확인 |
| 15:39 | `깃 허브에 올려줘` | 비공개 저장소 생성 후 push (모델 파일 포함) |
| 15:41 | `README.md 오늘 들은 수업 내용 정리해서 올려줘` | 수업 정리 README 작성 |
| 15:43 | `같이 올려주고 내용도 readme에 같이 정리해줘 그리고 readme에 사용한 프롬프트도 다 정리해줘` | 손·제스처 코드와 모델 추가, README에 실습 1·2와 이 프롬프트 목록 추가 |

### 프롬프트 작성 팁 (오늘 써 보고 느낀 점)

- **문서 URL + 원하는 결과**만 줘도 됩니다. "해당하는 라이브러리 받고, 파이썬, webcam 기반"처럼 언어와 입력 소스를 같이 적으면 원하는 형태로 나옵니다.
- 한 번 만든 뒤에는 "**여기서 또 해줘**" + 새 URL만으로 같은 방식의 코드를 만들 수 있습니다.
- 에러가 나면 "웹캠이 안 열려"처럼 **보이는 증상만** 말해도 Claude가 직접 실행해 보고 원인을 찾습니다.

---

## 참고 문서

- Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Gesture Recognizer: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
- 모델 라이선스: Apache 2.0 (Google)
