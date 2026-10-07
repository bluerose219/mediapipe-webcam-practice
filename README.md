# MediaPipe Face Landmarker - Webcam

MediaPipe Face Landmarker로 웹캠 영상에서 얼굴 랜드마크(478점)와 blendshape 점수를 실시간으로 검출합니다.

## 설치

```
pip install -r requirements.txt
```

## 실행

```
python face_landmarker_webcam.py
```

종료: `q` 또는 `ESC`

## 기능

- 얼굴 최대 2개 검출, 거울 모드
- 메시, 얼굴 윤곽, 입술, 눈, 눈썹, 홍채를 색깔별로 표시
- 첫 번째 얼굴의 blendshape 상위 5개 점수 표시 (예: `jawOpen`, `eyeBlinkLeft`)
- FPS, 검출된 얼굴 수 표시

## 참고

- 모델: [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) (Google, Apache 2.0)
- 경로에 한글 등 비ASCII 문자가 있으면 MediaPipe가 모델 파일을 열지 못하므로, 파일을 직접 읽어 `model_asset_buffer`로 전달합니다.
- 웹캠이 안 열리면 `cv2.VideoCapture(0, ...)`의 `0`을 `1`로 바꾸거나, 카메라를 쓰는 다른 프로그램을 종료하세요.
- 문서: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
