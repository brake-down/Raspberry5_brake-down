# GPT

import sys, json, time, threading
from collections import deque
from typing import Dict, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import sounddevice as sd

# ---------- openSMILE (pyopensmile ↔ opensmile 호환) ----------
try:
    import pyopensmile as opensmile
except ImportError:
    import opensmile  # type: ignore

# ===================== 설정 =====================
NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC", "USB")
SAMPLE_RATE = 48000
CHANNELS = 1
BUFFER_SECONDS = 3
FRAME_HZ = 4            # 2 → 4~8 정도 권장 (라즈피 성능에 맞춰)
BLOCKSIZE = 4096
SILENCE_RMS_THRESH = 3e-4  # 살짝 올려서 무음 억제
TARGET_RMS = 0.05          # AGC 목표 RMS
MAX_GAIN = 20.0

THRESH = {
    "F0_stddevNorm_high": 0.25,   # 실측으로 보정 권장
    "F0_risingSlope_high": 0.7,
    "F0_fallingSlope_high": -0.5,
    "jitter_high": 0.015,
    "shimmer_dB_high": 0.4,
    "HNR_low": 10.0,
}

FLAG_WEIGHTS = {
    "F0 변화 폭이 큼": 0.35,
    "F0 상승/하강 기울기 큼": 0.15,  # 상승/하강 둘 다 걸리면 2회 가산
    "Jitter 증가(떨림)": 0.10,
    "Shimmer 증가(음량 미세 변동)": 0.10,
    "HNR 낮음(잡음↑)": 0.10,
}

def pick_input_device(name_hints=NAME_HINTS) -> Optional[int]:
    try:
        devs = sd.query_devices()
        # 힌트 매칭
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                nm = d.get("name", "")
                if any(h in nm.upper() for h in name_hints):
                    print(f"[info] Found device by hint: '{nm}'")
                    return i
        # 기본 장치
        try:
            di = sd.default.device
            if isinstance(di, (tuple, list)) and di[0] is not None and di[0] != -1:
                return int(di[0])
        except Exception:
            pass
        # 첫 입력 장치
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                print(f"[info] Using first input: '{d.get('name','')}'")
                return i
    except Exception as e:
        print(f"[error] pick_input_device: {e}")
    return None

def safe_get(s: pd.Series, key: str) -> Optional[float]:
    try:
        return float(s[key]) if key in s else None
    except Exception:
        return None

def build_summary(func: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["F0_mean_semitone"]   = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_amean")
    out["F0_stddevNorm"]      = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    out["F0_meanRisingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")
    out["F0_meanFallingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope")
    out["jitterLocal_amean"]      = safe_get(func, "jitterLocal_sma3nz_amean")
    out["shimmerLocaldB_amean"]   = safe_get(func, "shimmerLocaldB_sma3nz_amean")
    out["HNRdBACF_amean"]         = safe_get(func, "HNRdBACF_sma3nz_amean")
    out["loudness_amean"]         = safe_get(func, "loudness_sma3_amean")

    flags = []
    if (out.get("F0_stddevNorm") or 0) >= THRESH["F0_stddevNorm_high"]:
        flags.append("F0 변화 폭이 큼")

    rs = out.get("F0_meanRisingSlope"); fs = out.get("F0_meanFallingSlope")
    if rs is not None and abs(rs) >= abs(THRESH["F0_risingSlope_high"]):
        flags.append("F0 상승/하강 기울기 큼")
    if fs is not None and abs(fs) >= abs(THRESH["F0_fallingSlope_high"]):
        flags.append("F0 상승/하강 기울기 큼")

    if (out.get("jitterLocal_amean") or 0) >= THRESH["jitter_high"]:
        flags.append("Jitter 증가(떨림)")
    if (out.get("shimmerLocaldB_amean") or 0) >= THRESH["shimmer_dB_high"]:
        flags.append("Shimmer 증가(음량 미세 변동)")
    hnr = out.get("HNRdBACF_amean")
    if hnr is not None and hnr <= THRESH["HNR_low"]:
        flags.append("HNR 낮음(잡음↑)")

    out["flags"] = flags
    comments = []
    if any("F0" in f for f in flags):
        comments.append("피치 변동이 커 당황/비명 가능성.")
    if any(("Jitter" in f) or ("Shimmer" in f) or ("HNR" in f) for f in flags):
        comments.append("발성 안정성이 낮아 긴장/불안정 음성일 수 있음.")
    out["summary_ko"] = " ".join(comments) if comments else "뚜렷한 이상 플래그 없음."
    return out

def build_probability(flags: list[str]) -> float:
    score = 0.0
    for f in flags:
        score += FLAG_WEIGHTS.get(f, 0.0)  # 상승/하강은 2회 가산될 수 있음
        
    SCALE = 0.6                 # 0.5~0.8 권장
    score *= SCALE
    return float(min(max(score, 0.0), 1.0))

class RealtimeVoiceMeter:
    """
    실시간 오디오 -> openSMILE(GeMAPSv01b Functionals) -> 규칙->확률
    - on_update: Callable[[dict], None]  # {"ts", "prob", "summary", "func_row"}
    - read(): 최신 측정 dict 반환(없으면 None)
    """
    def __init__(
        self,
        frame_hz: int = FRAME_HZ,
        ema_alpha: float = 0.3,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.frame_interval = 1.0 / float(frame_hz)
        self.ema_alpha = float(ema_alpha)
        self.on_update = on_update

        self.buf = deque(maxlen=int(SAMPLE_RATE * BUFFER_SECONDS))
        self.buf_lock = threading.Lock()

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.running = False
        self.stream = None
        self.thread = None

        self.latest: Optional[Dict[str, Any]] = None
        self.latest_lock = threading.Lock()
        self._ema: Optional[float] = None

    # --- audio I/O ---
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # 콜백에서 과도한 print는 피하기
            print(status, file=sys.stderr)
        with self.buf_lock:
            self.buf.extend(indata[:, 0])

    def _process_once(self):
        audio = None
        with self.buf_lock:
            if len(self.buf) == self.buf.maxlen:
                audio = np.array(self.buf, dtype=np.float32)

        if audio is None:
            return

        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < SILENCE_RMS_THRESH:
            return

        # 간단 AGC
        gain = min(MAX_GAIN, TARGET_RMS / max(rms, 1e-9))
        audio = np.clip(audio * gain, -1.0, 1.0)

        # openSMILE -> 기능행
        feats = self.smile.process_signal(audio, SAMPLE_RATE)
        if feats.empty:
            return
        row = feats.iloc[0]
        summary = build_summary(row)
        prob = build_probability(summary.get("flags", []))  # 0~1
        # EMA
        self._ema = self._ema * (1 - self.ema_alpha) + prob * self.ema_alpha if self._ema is not None else prob
        prob_smoothed = float(self._ema)

        out = {
            "ts": time.time(),
            "prob": prob_smoothed,      # 0~1
            "summary": summary,         # dict
            "func_row": row,            # pandas.Series (원하면 제거 가능)
        }
        with self.latest_lock:
            self.latest = out
        if self.on_update:
            try:
                self.on_update(out)
            except Exception as e:
                print(f"[warn] on_update error: {e}")

    def _loop(self):
        next_t = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now >= next_t:
                self._process_once()
                next_t += self.frame_interval
            time.sleep(0.001)

    def start(self):
        if self.running: return
        dev = pick_input_device()
        if dev is None:
            print("[error] no input device", file=sys.stderr)
            return

        self.running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=dev,
            blocksize=BLOCKSIZE,
            dtype="float32",
            latency="high",
            callback=self._audio_callback,
        )
        self.stream.start()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[info] realtime meter started (dev={dev}, frame_hz={1/self.frame_interval:.1f})")

    def stop(self):
        if not self.running: return
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
        print("[info] realtime meter stopped")

    def read(self) -> Optional[Dict[str, Any]]:
        with self.latest_lock:
            return None if self.latest is None else dict(self.latest)

# 단독 실행용
def main():
    meter = RealtimeVoiceMeter(frame_hz=FRAME_HZ)
    meter.start()
    try:
        while True:
            m = meter.read()
            if m:
                p = m["prob"]
                s = m["summary"]
                print(f"확률(EMA): {p*100:.1f}% | flags={s.get('flags', [])} | loud={s.get('loudness_amean')}")
            else:
                print("(warming up / silence)")
            time.sleep(1.0 / FRAME_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        meter.stop()

if __name__ == "__main__":
    main()
