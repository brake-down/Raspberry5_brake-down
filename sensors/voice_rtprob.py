# sensors/voice_rtprob.py
import sys, time, threading
from collections import deque
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import sounddevice as sd

# pyopensmile ↔ opensmile 호환 임포트
try:
    import pyopensmile as opensmile
except ImportError:
    import opensmile  # type: ignore

# ===== 기본 설정(필요 시 main에서 생성자 인수로 덮어쓰기) =====
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_BUFFER_SECONDS = 3
DEFAULT_FRAME_HZ = 4
DEFAULT_BLOCKSIZE = 4096
DEFAULT_SILENCE_RMS_THRESH = 3e-4
DEFAULT_TARGET_RMS = 0.05
DEFAULT_MAX_GAIN = 20.0
NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC", "USB")

# ===== 규칙/가중치 =====
THRESH = {
    "F0_stddevNorm_high": 0.25,
    "F0_risingSlope_high": 0.7,
    "F0_fallingSlope_high": -0.5,
    "jitter_high": 0.015,
    "shimmer_dB_high": 0.4,
    "HNR_low": 10.0,
}
FLAG_WEIGHTS = {
    "F0 변화 폭이 큼": 0.35,
    "F0 상승/하강 기울기 큼": 0.15,   # 상승/하강 두 번 걸리면 2회 가산
    "Jitter 증가(떨림)": 0.10,
    "Shimmer 증가(음량 미세 변동)": 0.10,
    "HNR 낮음(잡음↑)": 0.10,
}

SCALE = 0.6  # 최종 점수 스케일(0.5~0.8 권장)

def _pick_input_device(name_hints=NAME_HINTS) -> Optional[int]:
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                nm = d.get("name", "")
                if any(h in nm.upper() for h in name_hints):
                    print(f"[info] Found device by hint: '{nm}'")
                    return i
        di = sd.default.device
        if isinstance(di, (tuple, list)) and di and di[0] not in (None, -1):
            return int(di[0])
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                print(f"[info] Using first input: '{d.get('name','')}'")
                return i
    except Exception as e:
        print(f"[error] pick_input_device: {e}", file=sys.stderr)
    return None

def _safe_get(s: pd.Series, key: str) -> Optional[float]:
    try:
        return float(s[key]) if key in s else None
    except Exception:
        return None

def _build_summary(func: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["F0_mean_semitone"]    = _safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_amean")
    out["F0_stddevNorm"]       = _safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    out["F0_meanRisingSlope"]  = _safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")
    out["F0_meanFallingSlope"] = _safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope")
    out["jitterLocal_amean"]   = _safe_get(func, "jitterLocal_sma3nz_amean")
    out["shimmerLocaldB_amean"]= _safe_get(func, "shimmerLocaldB_sma3nz_amean")
    out["HNRdBACF_amean"]      = _safe_get(func, "HNRdBACF_sma3nz_amean")
    out["loudness_amean"]      = _safe_get(func, "loudness_sma3_amean")

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
    if any("F0" in f for f in flags): comments.append("피치 변동↑")
    if any(("Jitter" in f) or ("Shimmer" in f) or ("HNR" in f) for f in flags): comments.append("발성 안정성↓")
    out["summary_ko"] = " / ".join(comments) if comments else "뚜렷한 이상 없음"
    return out

def _flags_to_prob(flags: list[str]) -> float:
    score = 0.0
    for f in flags:
        score += FLAG_WEIGHTS.get(f, 0.0)  # 상승/하강은 중복 가능
    return float(min(max(score * SCALE, 0.0), 1.0))

class VoiceRTProb:
    """
    실시간: 마이크 → openSMILE(Functionals) → 요약/플래그 → 확률(0~1)
    - start()/stop()
    - read() -> {"ts","prob","summary","func_row"}
    - get_prob(default=0.0) -> float
    """
    def __init__(
        self,
        samplerate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        buffer_seconds: int = DEFAULT_BUFFER_SECONDS,
        frame_hz: int = DEFAULT_FRAME_HZ,
        blocksize: int = DEFAULT_BLOCKSIZE,
        silence_rms_thresh: float = DEFAULT_SILENCE_RMS_THRESH,
        target_rms: float = DEFAULT_TARGET_RMS,
        max_gain: float = DEFAULT_MAX_GAIN,
        device_index: Optional[int] = None,
        ema_alpha: float = 0.3,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug: bool = False,
    ):
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.buffer_seconds = int(buffer_seconds)
        self.frame_interval = 1.0 / float(frame_hz)
        self.blocksize = int(blocksize)
        self.silence_rms_thresh = float(silence_rms_thresh)
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.device_index = device_index
        self.ema_alpha = float(ema_alpha)
        self.on_update = on_update
        self.debug = debug

        self._buf = deque(maxlen=self.samplerate * self.buffer_seconds)
        self._buf_lock = threading.Lock()

        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None

        self._latest: Optional[Dict[str, Any]] = None
        self._latest_lock = threading.Lock()
        self._ema: Optional[float] = None

    # ========== public API ==========
    def start(self) -> None:
        if self._running:
            return
        dev = self.device_index if self.device_index is not None else _pick_input_device()
        if dev is None:
            print("[error] no input device", file=sys.stderr)
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=dev,
            blocksize=self.blocksize,
            dtype="float32",
            latency="high",
            callback=self._audio_callback,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.debug:
            print(f"[info] VoiceRTProb started (dev={dev}, hz={1/self.frame_interval:.1f})")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self.debug:
            print("[info] VoiceRTProb stopped")

    def read(self) -> Optional[Dict[str, Any]]:
        with self._latest_lock:
            return None if self._latest is None else dict(self._latest)

    def get_prob(self, default: float = 0.0) -> float:
        m = self.read()
        return float(m["prob"]) if m and ("prob" in m) else float(default)

    # ========== internals ==========
    def _audio_callback(self, indata, frames, time_info, status):
        if status and self.debug:
            print(status, file=sys.stderr)
        with self._buf_lock:
            self._buf.extend(indata[:, 0])

    def _process_once(self):
        audio = None
        with self._buf_lock:
            if len(self._buf) == self._buf.maxlen:
                audio = np.array(self._buf, dtype=np.float32)
        if audio is None or not audio.size:
            return

        # 무음 억제
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.silence_rms_thresh:
            return

        # 간단 AGC
        gain = min(self.max_gain, self.target_rms / max(rms, 1e-9))
        audio = np.clip(audio * gain, -1.0, 1.0)

        # 특징 추출 → 요약/플래그 → 확률
        feats = self._smile.process_signal(audio, self.samplerate)
        if feats.empty:
            return
        row = feats.iloc[0]
        summary = _build_summary(row)
        prob_raw = _flags_to_prob(summary.get("flags", []))

        # EMA 스무딩
        self._ema = self._ema * (1 - self.ema_alpha) + prob_raw * self.ema_alpha if self._ema is not None else prob_raw
        prob_smoothed = float(self._ema)

        out = {"ts": time.time(), "prob": prob_smoothed, "summary": summary, "func_row": row}
        with self._latest_lock:
            self._latest = out
        if self.on_update:
            try:
                self.on_update(out)
            except Exception as e:
                if self.debug:
                    print(f"[warn] on_update error: {e}", file=sys.stderr)

    def _loop(self):
        next_t = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now >= next_t:
                self._process_once()
                next_t += self.frame_interval
            time.sleep(0.001)
