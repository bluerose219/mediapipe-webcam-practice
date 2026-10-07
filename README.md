# MediaPipe 웹캠 실습 정리 (2026-10-07)

오늘 수업에서는 Google MediaPipe Tasks의 비전 모델 3가지를 웹캠에 연결해 실시간으로 돌려 보고, 이어서 **내가 정한 손 제스처를 직접 학습**시켜 **웹 페이지에서 제스처에 따라 불꽃·불·꽃 효과**가 나오게 만들었습니다.

| 순서 | 실습 | 실행 코드 | 모델 파일 | 검출 결과 |
|---|---|---|---|---|
| 1 | Hand Landmarker | `hand_webcam.py` | `hand_landmarker.task` | 손 21개 랜드마크 + 왼손/오른손 |
| 2 | Gesture Recognizer | `gesture_webcam.py` | `gesture_recognizer.task` | 손 랜드마크 + 제스처 이름 |
| 3 | Face Landmarker | `face_landmarker_webcam.py` | `face_landmarker.task` | 얼굴 478개 랜드마크 + blendshape |
| 4 | 커스텀 제스처 학습 | `collect_gestures.py` → `train_gestures.py` → `custom_gesture_webcam.py` | `models/custom_gesture.joblib` | 내가 만든 제스처 (heart, none, ok, rock) |
| 5 | 웹 제스처 이펙트 | `web/index.html` | `web/gesture_model.json` | ok → 불꽃놀이, rock → 손끝 불, heart → 꽃 |

## 파일 구성

```
├── hand_webcam.py              # 실습 1: 손 랜드마크
├── hand_landmarker.task        #   └ 모델 (7.8MB)
├── gesture_webcam.py           # 실습 2: 손 제스처 인식
├── gesture_recognizer.task     #   └ 모델 (8.4MB)
├── face_landmarker_webcam.py   # 실습 3: 얼굴 랜드마크
├── face_landmarker.task        #   └ 모델 (3.7MB)
│
├── gesture_common.py           # 실습 4: 공통 함수 (손 모델 불러오기, 특징 변환, 그리기)
├── collect_gestures.py         #   ├ ① 데이터 수집
├── train_gestures.py           #   ├ ② 학습
├── custom_gesture_webcam.py    #   └ ③ 추론 (Python 웹캠)
├── data/gestures.csv           #   학습 데이터 (heart 759, none 653, ok 614, rock 614개)
├── models/custom_gesture.joblib  # 학습된 모델
│
├── export_web_model.py         # 실습 5: 학습된 모델 → 웹용 JSON 변환
├── web/
│   ├── index.html              #   웹 페이지 (제스처 이펙트)
│   ├── gesture_model.json      #   웹용 모델 가중치
│   └── hand_landmarker.task    #   손 인식 모델 (복사본)
│
├── requirements.txt
└── README.md
```

## 빠른 실행

```
pip install -r requirements.txt
python hand_webcam.py              # 손 랜드마크
python gesture_webcam.py           # 손 제스처
python face_landmarker_webcam.py   # 얼굴 랜드마크
python custom_gesture_webcam.py    # 내가 학습한 제스처

python -m http.server 8000 -d web  # 웹 이펙트 → Chrome에서 http://localhost:8000
```

Python 프로그램은 모두 `q` 또는 `ESC`로 종료합니다. 모델 파일과 학습된 모델이 저장소에 들어 있으므로 따로 받거나 학습할 필요 없이 바로 실행됩니다.

---

## 1. 환경 준비

```
pip install -r requirements.txt   # mediapipe, opencv-python, scikit-learn, joblib
```

- 버전: Python 3.14, mediapipe 1.1.0, opencv-python 5.0.0, scikit-learn 1.9.1 / 웹: `@mediapipe/tasks-vision` 1.1.0 (CDN)
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

## 6. 실습 4 — 커스텀 제스처 학습

기본 Gesture Recognizer는 정해진 7가지 제스처만 인식합니다. 내가 원하는 제스처(🫶 heart, 👌 ok, ✊ rock 등)를 인식하게 하려면 직접 데이터를 모아 학습해야 합니다.

### 왜 Model Maker 대신 scikit-learn을 썼나

- Google 공식 학습 도구 **MediaPipe Model Maker**는 TensorFlow가 필요한데, **Python 3.14용 TensorFlow가 없어서** 설치가 안 됩니다.
- 대신 **Hand Landmarker로 손 관절 21개 좌표를 뽑고**, 그 좌표 모양을 **scikit-learn 신경망(MLP)** 으로 분류했습니다.
- 이미지가 아니라 좌표 63개(21개 × x, y, z)만 쓰기 때문에 데이터가 적어도 잘 되고, 학습도 몇 초면 끝납니다.

```
웹캠 → Hand Landmarker → 손 좌표 21개 → 특징 63개로 정규화 → MLP 분류기 → "heart" (0.95)
```

### 특징 정규화 (`gesture_common.landmarks_to_features`)

같은 제스처라면 손이 화면 어디에 있든, 카메라에서 얼마나 떨어져 있든, 왼손이든 오른손이든 같은 값이 나오도록 바꿉니다.

```python
pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
pts -= pts[0]                          # ① 손목(0번)을 원점으로 → 화면 속 위치와 무관
if handedness == "Left":
    pts[:, 0] = -pts[:, 0]             # ② 왼손은 좌우 뒤집기 → 한 손으로 모아도 양손 인식
scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
pts /= scale                           # ③ 손 크기로 나누기 → 카메라와의 거리와 무관
features = pts.flatten()               # 63개
```

### ① 데이터 수집

```
python collect_gestures.py none heart ok rock
```

| 키 | 동작 |
|---|---|
| `1`~`9` | 수집할 제스처 선택 (인자 순서대로) |
| `SPACE` | 녹화 시작/정지 (녹화 중엔 손이 빨갛게 표시되고, 손이 보이는 프레임마다 저장) |
| `q` / `ESC` | 종료 |

- 결과는 `data/gestures.csv`에 한 줄에 `제스처 이름, 특징 63개`로 저장됩니다. 여러 번 실행하면 이어서 저장됩니다.
- **`none` 클래스를 꼭 만드세요.** 아무 제스처도 아닌 평소 손 모양을 모아 두면 엉뚱한 오인식이 줄어듭니다.
- 제스처마다 **200~300개 이상** 모으세요. 녹화 중에 손을 돌리고, 앞뒤로 움직이고, 화면 여기저기로 옮기면 인식이 더 잘 됩니다.
- OpenCV는 화면에 한글을 못 그리므로 제스처 이름은 영어로 씁니다.

### ② 학습

```
python train_gestures.py
```

- 구조: `StandardScaler`(값 크기 맞추기) → `MLPClassifier`(은닉층 128 → 64, ReLU)
- 데이터의 20%를 떼어 **정확도, 클래스별 성능, 혼동 행렬**(어떤 제스처끼리 헷갈렸는지)을 출력합니다.
- 확인 후 전체 데이터로 다시 학습해서 `models/custom_gesture.joblib`에 저장합니다.
- 정확도가 낮거나 헷갈리는 짝이 있으면 그 제스처를 더 모으고 다시 학습합니다.

### ③ 추론

```
python custom_gesture_webcam.py
```

- 손 위에 `Right: heart (0.95)`처럼 표시됩니다.
- 확률이 `THRESHOLD`(0.7)보다 낮으면 `?`로 표시합니다.

## 7. 실습 5 — 웹 제스처 이펙트

학습한 모델을 브라우저에서 돌려서, 제스처에 따라 화면에 효과가 나오게 했습니다.

| 제스처 | 효과 |
|---|---|
| 👌 **ok** | 손 위치와 하늘에서 불꽃놀이가 터짐 (0.45초마다) |
| ✊ **rock** | 다섯 손가락 끝에서 불이 피어오름 (매 프레임) |
| 🫶 **heart** | 손 주변에 꽃이 피고 꽃잎이 흩날림 |

### 실행

```
python export_web_model.py               # 모델을 웹용으로 변환 (다시 학습했으면 매번 실행)
python -m http.server 8000 -d web        # 웹 서버 켜기
```

Chrome에서 **http://localhost:8000** 을 열고 카메라 권한을 허용합니다.

> `index.html`을 더블클릭해서 `file://`로 열면 보안 정책 때문에 카메라·모델 불러오기가 안 됩니다. 꼭 `localhost`로 여세요.

### 동작 원리

1. **모델 변환** (`export_web_model.py`): sklearn 모델에서 정규화 값(`mean`, `scale`)과 신경망 가중치(`coefs_`, `intercepts_`)를 꺼내 `gesture_model.json`으로 저장
2. **손 인식**: 브라우저용 MediaPipe(`@mediapipe/tasks-vision`)의 `HandLandmarker`를 CDN에서 불러와 사용
3. **제스처 분류**: JSON 가중치로 신경망 계산(행렬 곱 → ReLU → softmax)을 JavaScript로 직접 구현. Python 결과와 소수 넷째 자리까지 같음을 확인
4. **효과**: `<canvas>` 파티클 애니메이션
   - 불꽃놀이: 한 점에서 원형으로 퍼지는 파티클 + 중력 + 잔상
   - 손끝 불: 손가락 끝(4, 8, 12, 16, 20번 랜드마크)에서 위로 오르며 흰색 → 노랑 → 주황 → 빨강으로 식는 불씨
   - 꽃: 꽃잎 5~7장짜리 꽃이 커지며 피어나고, 꽃잎이 흩날림

```javascript
// 브라우저에서 sklearn MLP와 같은 계산 (index.html의 predict 함수)
let h = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);  // StandardScaler
model.weights.forEach((W, layer) => {
  const out = model.biases[layer].slice();
  for (let i = 0; i < h.length; i++)
    for (let j = 0; j < out.length; j++) out[j] += h[i] * W[i][j];     // h · W + b
  const last = layer === model.weights.length - 1;
  h = last ? out : out.map(v => Math.max(0, v));                        // 은닉층 ReLU
});
// 마지막에 softmax로 확률 계산
```

### 학습 때와 똑같이 맞춰야 하는 것

- **거울 모드**: Python 수집 코드는 `cv2.flip`으로 뒤집은 영상에서 좌표를 모았으므로, 웹에서도 영상을 좌우로 뒤집은 캔버스를 MediaPipe에 넣습니다. 안 그러면 왼손/오른손이 바뀌어 인식이 틀어집니다.
- **특징 정규화**: `landmarksToFeatures`(JS)는 Python의 `landmarks_to_features`와 똑같은 계산을 합니다.
- **오인식 방지**: 확률 70% 이상인 같은 제스처가 4프레임 연속 나와야 효과를 냅니다 (`THRESHOLD`, `STABLE_FRAMES`).

---

## 8. 문제 해결 (오늘 겪은 것)

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

### MediaPipe Model Maker가 설치되지 않음

- TensorFlow가 Python 3.14를 지원하지 않아서 `pip install tensorflow`가 실패합니다.
- 이 저장소처럼 손 좌표 + scikit-learn으로 학습하거나, Model Maker를 꼭 써야 하면 Python 3.11 같은 이전 버전 가상환경을 따로 만들어야 합니다.

### 무시해도 되는 로그

실행할 때 나오는 `WARNING: Logging before InitGoogle()`, `Created TensorFlow Lite XNNPACK delegate` 같은 메시지는 정상 동작 중에 나오는 로그입니다.

---

## 9. 사용한 프롬프트 (Claude Code)

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
| 15:43 | `같이 올려주고 내용도 readme에 같이 정리해줘 그리고 readme에 사용한 프롬프트도 다 정리해줘` | 손·제스처 코드와 모델 추가, README에 실습 1·2와 이 프롬프트 목록 추가 |

### 세션 2 (이어서) — 커스텀 제스처 학습, 웹 이펙트

| 시각 | 프롬프트 | 결과 |
|---|---|---|
| 16:02 | `제스쳐인식 코드가 잘 동작하지만 내가 원하는 제스쳐를 추가 학습해서 추론해 보고 싶다. 수집하는 코드와 훈련하는 코드를 가각 작성해주고 사용방법을 알려줘` | Model Maker가 Python 3.14에서 안 돼서 손 좌표 + scikit-learn 방식으로 `collect_gestures.py`, `train_gestures.py`, `custom_gesture_webcam.py` 작성 |
| 16:14 | `훈련한 모델로 rock 나오면 불꽃이 터지게 하고 heart가 나오면 꽃이 나오도록 웹버전으로 만들어 줘` | 모델을 JSON으로 내보내는 `export_web_model.py`와 `web/index.html` 작성, Chrome에서 Python과 예측값 일치 확인 |
| 16:18 | `ok 여기에 불꽃 넣고 rock은 손가락 끝에 불이 나오는 걸로 바꿔줘` | 불꽃놀이를 ok로 옮기고, rock은 손끝 불 효과로 변경 |
| 16:19 | `깃허브에 올리고 readme도 정리해줘` | 새 코드·데이터·모델 업로드, README에 실습 4·5 추가 |

### 프롬프트 작성 팁 (오늘 써 보고 느낀 점)

- **문서 URL + 원하는 결과**만 줘도 됩니다. "해당하는 라이브러리 받고, 파이썬, webcam 기반"처럼 언어와 입력 소스를 같이 적으면 원하는 형태로 나옵니다.
- 한 번 만든 뒤에는 "**여기서 또 해줘**" + 새 URL만으로 같은 방식의 코드를 만들 수 있습니다.
- 에러가 나면 "웹캠이 안 열려"처럼 **보이는 증상만** 말해도 Claude가 직접 실행해 보고 원인을 찾습니다.
- "rock 나오면 불꽃, heart 나오면 꽃"처럼 **조건 → 결과**로 말하면 원하는 동작이 정확히 나옵니다. 마음에 안 들면 "ok에 불꽃 넣고 rock은 손끝 불로"처럼 바꿀 부분만 말하면 됩니다.

---

## 참고 문서

- Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Gesture Recognizer: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
- Gesture Recognizer 커스터마이즈 (Model Maker): https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
- MediaPipe Tasks Vision (웹): https://www.npmjs.com/package/@mediapipe/tasks-vision
- 모델 라이선스: Apache 2.0 (Google)
