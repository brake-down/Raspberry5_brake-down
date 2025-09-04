# Raspberry5_brake-down

이 저장소는 라즈베리 파이 5에서 운전자의 급발진 또는 페달 오조작을 실시간으로 감지하기 위한 데모 시스템입니다. 오디오(음성), 비디오(표정), OBD-II 데이터를 동시에 수집하고, 간단한 규칙 기반 결합 로직으로 경고를 출력합니다.

## 전체 구조
- **프로듀서(입력)**: 마이크, 카메라, OBD-II 센서에서 데이터를 읽어 `queue.Queue`에 메시지를 넣습니다.
- **컨슈머(판단 루프)**: 큐의 데이터를 모아 `FusionState`에 저장하고, 최근 0.5초 창을 기준으로 이상 여부를 판정합니다.
- **출력**: 상태(`OK`, `WARNING`, `ALERT`)와 점수를 표준 출력으로 기록하고, `/dev/fb0` 프레임버퍼에 경고 화면을 표시합니다.

## 주요 모듈
- [`main.py`](main.py): 스레드를 띄워 각 프로듀서를 실행하고 `decision_loop`를 통해 최종 상태를 결정합니다.
- [`fusion.py`](fusion.py): 메시지 포맷, 구성값, OBD 비정상 규칙, 최종 의사결정 로직을 담고 있습니다.
- **producers/**
  - [`audio_rtprob_producer.py`](producers/audio_rtprob_producer.py): `VoiceRTProb` 센서에서 음성 놀람 확률을 읽어 큐에 넣습니다.
  - [`video_rtprob_producer.py`](producers/video_rtprob_producer.py): Picamera2와 FER 모델을 사용해 얼굴 놀람 확률을 계산합니다.
  - [`serial_real.py`](producers/serial_real.py): 실차 OBD-II 시리얼 데이터를 파싱하여 속도, RPM, 스로틀, 브레이크 값을 제공합니다.
  - 기타 시뮬레이터(`serial_loop.py`, `warning_*`)가 포함되어 테스트에 활용됩니다.
- **sensors/**
  - [`voice_rtprob.py`](sensors/voice_rtprob.py): openSMILE 기능 추출과 규칙 기반 가중치를 이용해 음성으로부터 스트레스 확률(0~1)을 산출합니다.
  - [`video_prob_rt_picam.py`](sensors/video_prob_rt_picam.py): Haar Cascade 얼굴 검출 후 TFLite FER 모델로 표정에서 놀람 확률(0~100%)을 계산합니다.
- [`utils/qput.py`](utils/qput.py): 큐가 가득 찼을 때 가장 오래된 항목을 버리고 새 항목을 넣는 보조 함수입니다.

## 동작 과정
1. 각 프로듀서가 `Msg` 객체를 큐에 비동기로 삽입합니다.
2. `decision_loop`가 큐를 주기적으로 비우고 `FusionState`에 데이터를 추가합니다.
3. `decide()` 함수가 OBD 이상 징후와 사람 반응의 피크/지속성을 기반으로 `OK`/`WARNING`/`ALERT`를 판정합니다.
4. 결과는 JSON 형태로 표준 출력에 기록되고, `ALERT`/`WARNING` 시 프레임버퍼에 경고 화면이 표시됩니다.

## 실행 예시
```bash
pip install -r requirements.txt
python main.py
```
Picamera2, 마이크(예: RØDE VideoMic GO II), OBD-II 시리얼 장치가 연결된 라즈베리 파이에서 실행하는 것을 전제로 합니다.

## 참고
- `scripts/`와 `openSmile/` 폴더에는 실험용 스크립트와 초기 분석 도구가 포함되어 있습니다.

