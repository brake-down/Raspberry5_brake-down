# realtime_analyzer.py
# Raspberry Pi 5 + RØDE VideoMic GO II 실시간 openSMILE 분석기
# - 장치 자동 선택(RØDE 우선)
# - 48kHz/모노 고정
# - 버퍼 워밍업 + 무음 게이트로 NaN 억제
# - GeMAPS(Functionals) 지표 출력

import sys
import time
import threading
from collections import deque
from typing import Dict, Optional

import numpy as np
import sounddevice as sd

# ---------- openSMILE 파이썬 패키지 임포트 (pyopensmile ↔ opensmile 호환) ----------
try:
    import pyopensmile as pyopensmile  # 셔틀 파일(pyopensmile.py)이 있으면 이 경로로
except Exception:
    try:
        import opensmile as pyopensmile
    except Exception as e:
        raise RuntimeError(
            "openSMILE python package가 없습니다. 가상환경에서 `pip install opensmile` 후 실행하세요."
        ) from e

# ---------- 기본 설정 (필요시 바꿔도 됨) ----------
NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC")
SAMPLE_RATE = 48000          # RØDE는 48kHz 권장
CHANNELS = 1                 # 모노
BUFFER_SECONDS = 8         # 슬라이딩 윈도우 길이(초) — 짧으면 NaN 날 수 있음
FRAME_HZ = 3              # 초당 처리 횟수
BLOCKSIZE = 4096          # 오디오 콜백 블록 크기
SILENCE_RMS_THRESH = 3e-4    # 무음 판정 임계치(너무 낮으면 NaN 가능)

# ---------- 입력 장치 선택 ----------
def pick_input_device(name_hints=NAME_HINTS) -> Optional[int]:
    devs = sd.query_devices()
    # 힌트 매칭 우선
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            nm = d.get("name", "")
            if any(h in nm.upper() for h in name_hints):
                return i
    # 없으면 첫 입력 장치
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i
    return None

INPUT_INDEX = pick_input_device()

# sounddevice 기본값 고정
sd.default.channels = CHANNELS
sd.default.samplerate = SAMPLE_RATE
if INPUT_INDEX is not None:
    sd.default.device = (INPUT_INDEX, None)  # (input, output)

class RealTimeOpenSmile:
    """
    마이크에서 실시간 오디오를 읽어 BUFFER_SECONDS 길이로 슬라이딩 버퍼를 유지하고,
    FRAME_HZ 주기로 openSMILE GeMAPS(Functionals) 특징을 출력.
    """

    def __init__(
        self,
        device_index: Optional[int] = INPUT_INDEX,
        samplerate: int = SAMPLE_RATE,
        buffer_seconds: int = BUFFER_SECONDS,
        frame_hz: int = FRAME_HZ,
    ) -> None:
        self.device = device_index
        self.samplerate = int(samplerate)
        self.buffer_samples = int(self.samplerate * buffer_seconds)
        self.buffer: deque[float] = deque(maxlen=self.buffer_samples)
        self.lock = threading.Lock()
        self.frame_interval = 1.0 / float(frame_hz)

        # GeMAPS v01b 기능셋(Functional) — 가볍고 실시간에 적합
        self.smile = pyopensmile.Smile(
            feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
            feature_level=pyopensmile.FeatureLevel.Functionals,
        )

        # 보고할 지표 매핑(해당 키가 실제로 제공될 때만 사용)
        feature_map = {
            "loudness": "loudness_sma3_amean",
            "pitch":    "F0semitoneFrom27.5Hz_sma3nz_amean",
            "jitter":   "jitterLocal_sma3nz_amean",
            "shimmer":  "shimmerLocaldB_sma3nz_amean",
            "hnr":      "HNRdBACF_sma3nz_amean",
        }
        available = set(self.smile.feature_names)
        self.features: Dict[str, str] = {n: k for n, k in feature_map.items() if k in available}

    # 오디오 콜백: 모노 첫 채널만 저장
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            self.buffer.extend(indata[:, 0])

    # 버퍼 처리 → openSMILE로 특징 계산
    def _process_buffer(self) -> None:
        with self.lock:
            if len(self.buffer) < self.buffer_samples:
                return  # 워밍업(버퍼가 아직 꽉 차지 않음)
            audio = np.array(self.buffer, dtype=np.float32)

        # 무음 게이트: 너무 작은 입력이면 건너뜀(NaN 방지)
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < SILENCE_RMS_THRESH:
            print(f"(silence) waiting... rms={rms:.6g}")
            return

        target = 0.05
        gain = min(20.0, target / max(rms, 1e-8))
        audio = np.clip(audio * gain, -1.0, 1.0)
        #audio = audio.reshape(-1, 1)  # (T, 1ch)
        feats = self.smile.process_signal(audio, self.samplerate)
        series = feats.iloc[0]
        values = {name: float(series.get(key, np.nan)) for name, key in self.features.items()}
        line = " | ".join(f"{n}: {v:.2f}" if np.isfinite(v) else f"{n}: NaN" for n, v in values.items())
        print(line)

    def run(self) -> None:
        print(f"[info] device={self.device}, sr={self.samplerate}, ch={CHANNELS}")
        # 장치/레이트 유효성 점검(가능하면 미리 체크)
        try:
            sd.check_input_settings(
                device=self.device if self.device is not None else sd.default.device[0],
                channels=CHANNELS,
                samplerate=self.samplerate,
                dtype="float32",
            )
        except Exception as e:
            print("[warn] 입력 장치 설정 확인 중 경고:", e)

        stream_kwargs = dict(
            samplerate=self.samplerate,
            channels=CHANNELS,
            blocksize=BLOCKSIZE,
            dtype="float32",
            callback=self._audio_callback,
            latency='high',              # ★ 추가
        )
        if self.device is not None:
            stream_kwargs["device"] = self.device

        # 입력 스트림 열고 버퍼가 가득 찰 때까지 워밍업
        with sd.InputStream(**stream_kwargs):
            t0 = time.time()
            warmup_timeout = (self.buffer_samples / self.samplerate) * 2.0  # 버퍼 길이의 2배 시간까지 대기
            while True:
                with self.lock:
                    filled = len(self.buffer)
                if filled >= self.buffer_samples:
                    break
                if time.time() - t0 > warmup_timeout:
                    print("[warn] buffer did not fill in time; continuing anyway")
                    break
                time.sleep(0.05)

            next_time = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    if now >= next_time:
                        self._process_buffer()
                        next_time += self.frame_interval
                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nStopping stream...")

if __name__ == "__main__":
    analyzer = RealTimeOpenSmile(
        device_index=INPUT_INDEX,
        samplerate=SAMPLE_RATE,
        buffer_seconds=BUFFER_SECONDS,
        frame_hz=FRAME_HZ,
    )
    analyzer.run()
