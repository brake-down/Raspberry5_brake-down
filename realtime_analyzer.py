# realtime_analyzer.py
# Raspberry Pi 5 + RØDE VideoMic GO II 실시간 openSMILE 분석기 (모듈/단독 겸용)

import sys
import time
import threading
from collections import deque
from typing import Dict, Optional

import numpy as np
import sounddevice as sd

# ---------- openSMILE 파이썬 패키지 임포트 ----------
try:
    import pyopensmile as pyopensmile
except Exception:
    try:
        import opensmile as pyopensmile
    except Exception as e:
        raise RuntimeError(
            "openSMILE python package가 없습니다. 가상환경에서 `pip install opensmile` 후 실행하세요."
        ) from e

# ---------- 기본 설정 (튜닝) ----------
NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC", "USB")
SAMPLE_RATE = 48000
CHANNELS = 1
BUFFER_SECONDS = 2          # ↓ 8s -> 2s (워밍업 짧게)
FRAME_HZ = 5                # ↓ 3 -> 5 (반응성 ↑)
BLOCKSIZE = 2048            # ↓ 4096 -> 2048 (오버플로우 완화)
SILENCE_RMS_THRESH = 5e-5   # ↓ 3e-4 -> 5e-5 (무음 임계 완화)
RMS_WINDOW_SEC = 0.5        # 무음 판정은 최근 0.5s만 사용

def _device_name(idx: Optional[int]) -> str:
    try:
        if idx is None:
            return "(default)"
        return sd.query_devices(idx)["name"]
    except Exception:
        return "(unknown)"

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

# 기본값(가능하면 고정)
sd.default.channels = CHANNELS
sd.default.samplerate = SAMPLE_RATE
if INPUT_INDEX is not None:
    # (input, output) 튜플로 설정. 출력은 유지.
    try:
        cur = sd.default.device
        outdev = None
        if isinstance(cur, (list, tuple)) and len(cur) == 2:
            outdev = cur[1]
        sd.default.device = (INPUT_INDEX, outdev)
    except Exception:
        sd.default.device = (INPUT_INDEX, None)

class RealTimeOpenSmile:
    """
    - start()/stop(): 백그라운드 스레드에서 마이크 읽고 최신 특징 보관
    - read(): 최신 특징 dict 반환 (없으면 None)
    - run(): 단독 실행 시 콘솔에 사람이 보기 좋은 줄 출력
    """
    def __init__(
        self,
        device_index: Optional[int] = INPUT_INDEX,
        samplerate: int = SAMPLE_RATE,
        buffer_seconds: int = BUFFER_SECONDS,
        frame_hz: int = FRAME_HZ,
        silence_rms_thresh: float = SILENCE_RMS_THRESH,
        blocksize: int = BLOCKSIZE,
        verbose_warnings: bool = True,
    ) -> None:
        self.device = device_index
        self.samplerate = int(samplerate)
        self.buffer_samples = int(self.samplerate * buffer_seconds)
        self._buf: deque[float] = deque(maxlen=self.buffer_samples)
        self._buf_lock = threading.Lock()

        self.frame_interval = 1.0 / float(frame_hz)
        self.silence_rms_thresh = float(silence_rms_thresh)
        self.blocksize = int(blocksize)
        self.verbose_warnings = verbose_warnings

        # openSMILE: GeMAPS Functionals
        self.smile = pyopensmile.Smile(
            feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
            feature_level=pyopensmile.FeatureLevel.Functionals,
        )

        self._latest: Optional[Dict[str, float]] = None
        self._latest_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -------------- Public API --------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    def read(self) -> Optional[Dict[str, float]]:
        with self._latest_lock:
            return None if self._latest is None else dict(self._latest)

    # -------------- Internal: audio I/O --------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status and self.verbose_warnings:
            # status에 input overflow 표시가 들어옴
            print(status, file=sys.stderr)
        with self._buf_lock:
            self._buf.extend(indata[:, 0])

    def _extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        prefs = {
            "loudness": ["loudness_sma3nz_amean", "loudness_sma3_amean"],
            "pitch":    ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0final_sma_amean"],
            "jitter":   ["jitterLocal_sma3nz_amean", "jitterLocal_sma_amean"],
            "shimmer":  ["shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma_amean"],
            "hnr":      ["HNRdBACF_sma3nz_amean", "HNRdBACF_sma_amean"],
        }
        feats = self.smile.process_signal(audio, self.samplerate)
        series = feats.iloc[0]
        def first_present(keys):
            for k in keys:
                if k in series:
                    try:
                        return float(series[k])
                    except Exception:
                        return float("nan")
            return float("nan")
        return {name: first_present(keys) for name, keys in prefs.items()}

    # -------------- Internal: main loop --------------
    def _loop(self):
        # 입력 설정 점검
        try:
            sd.check_input_settings(
                device=self.device if self.device is not None else sd.default.device[0],
                channels=CHANNELS,
                samplerate=self.samplerate,
                dtype="float32",
            )
        except Exception as e:
            if self.verbose_warnings:
                print("[warn] 입력 장치 설정 확인 중 경고:", e)

        stream_kwargs = dict(
            samplerate=self.samplerate,
            channels=CHANNELS,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._audio_callback,
            latency="low",   # ← overflow 완화(콜백을 더 자주 불러줌)
        )
        if self.device is not None:
            stream_kwargs["device"] = self.device

        dev_name = _device_name(self.device)
        print(f"[info] using input device idx={self.device} name='{dev_name}' sr={self.samplerate} ch={CHANNELS}")

        with sd.InputStream(**stream_kwargs):
            # 워밍업 (버퍼 채우기) — 최대 2×버퍼길이 대기
            t0 = time.time()
            warmup_timeout = (self.buffer_samples / self.samplerate) * 2.0
            while self._running:
                with self._buf_lock:
                    filled = len(self._buf)
                if filled >= self.buffer_samples:
                    break
                if time.time() - t0 > warmup_timeout:
                    if self.verbose_warnings:
                        print("[warn] buffer did not fill in time; continuing anyway")
                    break
                time.sleep(0.02)

            # 주기 처리 루프
            next_time = time.monotonic()
            rms_tail = max(1, int(self.samplerate * RMS_WINDOW_SEC))
            while self._running:
                now = time.monotonic()
                if now >= next_time:
                    audio = None
                    with self._buf_lock:
                        if len(self._buf) >= self.buffer_samples:
                            audio = np.array(self._buf, dtype=np.float32)

                    if audio is not None and audio.size:
                        # 무음 게이트: 최근 0.5s 기준
                        tail = audio[-rms_tail:] if audio.size >= rms_tail else audio
                        rms = float(np.sqrt(np.mean(tail**2)))
                        # 디버깅용: 너무 조용할 때 RMS 보여주기 (원한다면 주석 해제)
                        # print(f"[debug] rms_last{RMS_WINDOW_SEC*1000:.0f}ms = {rms:.6g}")

                        if rms >= self.silence_rms_thresh:
                            # 가벼운 AGC
                            target = 0.05
                            gain = min(20.0, target / max(rms, 1e-8))
                            audio = np.clip(audio * gain, -1.0, 1.0)
                            feats = self._extract_features(audio)
                            with self._latest_lock:
                                self._latest = feats
                        # else: 무음이면 최신값 유지

                    next_time += self.frame_interval
                time.sleep(0.001)

    # -------------- Standalone runner --------------
    def run(self) -> None:
        print(f"[info] device={self.device} name='{_device_name(self.device)}', sr={self.samplerate}, ch={CHANNELS}")
        self.start()
        try:
            while True:
                m = self.read()
                if m:
                    print(
                        f"loudness: {m.get('loudness', float('nan')):.2f} | "
                        f"pitch: {m.get('pitch', float('nan')):.2f} | "
                        f"jitter: {m.get('jitter', float('nan')):.2f} | "
                        f"shimmer: {m.get('shimmer', float('nan')):.2f} | "
                        f"hnr: {m.get('hnr', float('nan')):.2f}"
                    )
                else:
                    print("(warming up / silence)")
                time.sleep(1.0 / FRAME_HZ)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

# -------------- Script entry --------------
if __name__ == "__main__":
    analyzer = RealTimeOpenSmile(
        device_index=INPUT_INDEX,
        samplerate=SAMPLE_RATE,
        buffer_seconds=BUFFER_SECONDS,
        frame_hz=FRAME_HZ,
    )
    analyzer.run()
