#!/usr/bin/env python3
"""
Real-time pedal misoperation detector (VIDEO/AUDIO/OBD2)
- AUDIO: openSMILE 실시간(있으면) / 없으면 랜덤 폴백
- VIDEO: 모의
- SERIAL: loop:// 가상(기본) 또는 실제 포트
- WARNING/ALERT만 JSON 출력
- 동시성 확인: sources_ts / sources_lag_ms / last_seq / sources_count_recent_1s / sources_src
"""

import time
import json
import math
import queue
import threading
import re
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

# ---------------- Config switches ----------------
FORCE_AUDIO_FALLBACK = False   # True로 두면 무조건 난수 폴백 사용(설치 전 빠른 테스트)

# -------- PySerial --------
import serial
try:
    from serial import serial_for_url as _serial_for_url
except Exception:
    _serial_for_url = None

# -------- (옵션) 오디오 실시간 분석 의존성 --------
HAVE_SMILE = True
try:
    import numpy as np
    import sounddevice as sd
    try:
        import pyopensmile as pyopensmile
    except Exception:
        import opensmile as pyopensmile
except Exception:
    HAVE_SMILE = False

if FORCE_AUDIO_FALLBACK:
    HAVE_SMILE = False

# =========================
# 0) 설정(튜닝 가능한 값들)
# =========================

@dataclass
class Config:
    # 버퍼/창
    buffer_sec: float = 2.0
    recent_window_sec: float = 0.5
    # 임계값
    TA: float = 0.5
    TV: float = 0.5
    persist_min_frames: int = 3
    # OBD 비정상 판단 임계
    throttle_brake_conflict_thr: float = 0.3
    throttle_high_thr: float = 0.8
    low_speed_thr: float = 2.0
    rpm_jump_thr: float = 1200.0
    speed_small_thr: float = 1.0
    # 쿨다운
    cooldown_sec: float = 2.0
    # 모의 입력 주기
    audio_hz: float = 20.0
    video_hz: float = 20.0
    obd_hz: float = 50.0
    # 로깅
    log_to_file: bool = False
    log_path: str = "events.log"

CFG = Config()

# ======================
# 1) 공통 메시지/큐 정의
# ======================

class MsgType(Enum):
    VIDEO = auto()
    AUDIO = auto()
    SERIAL = auto()

@dataclass
class Msg:
    type: MsgType
    ts: float
    data: Dict[str, Any]
    src: str = ""
    seq: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

Q: queue.Queue[Msg] = queue.Queue(maxsize=1024)
EV_STOP = threading.Event()

# =========================
# 2) 유틸: 버퍼/통계 계산기
# =========================

def now_s() -> float:
    return time.time()

def prune_buffer(buf: List[Msg], horizon: float) -> None:
    t = now_s()
    while buf and (t - buf[0].ts) > horizon:
        buf.pop(0)

def recent_values(buf: List[Msg], key: str, win: float) -> List[float]:
    t = now_s()
    return [m.data.get(key, 0.0) for m in buf if (t - m.ts) <= win]

def peak(buf: List[Msg], key: str, win: float) -> float:
    vals = recent_values(buf, key, win)
    return max(vals) if vals else 0.0

def count_over(buf: List[Msg], key: str, thr: float, win: float) -> int:
    vals = recent_values(buf, key, win)
    return sum(1 for v in vals if v > thr)

def delta_over_window(buf: List[Msg], key: str, win: float) -> float:
    t = now_s()
    arr = [(m.ts, float(m.data.get(key, 0.0))) for m in buf if (t - m.ts) <= win]
    if len(arr) < 2:
        return 0.0
    arr.sort(key=lambda x: x[0])
    return arr[-1][1] - arr[0][1]

# ===========================
# 3) Producers (입력 스레드)
# ===========================

# ---- VIDEO (모의) ----
def video_producer():
    period = 1.0 / CFG.video_hz
    seq = 0
    while not EV_STOP.is_set():
        val = max(0.0, min(1.0, random.gauss(0.12, 0.1)))
        if random.random() < 0.015:
            val = 0.65 + random.random()*0.35
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": val}, src="cam0", seq=seq))
        seq += 1
        time.sleep(period)

# ---- AUDIO (openSMILE 실시간 / 폴백 랜덤) ----
if HAVE_SMILE:
    NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC")
    SAMPLE_RATE = 48000
    CHANNELS = 1
    BUFFER_SECONDS = 2          # 8→2: 첫 결과 빨리
    FRAME_HZ = 5                # 3→5: 메시지 조금 더 자주
    BLOCKSIZE = 4096
    SILENCE_RMS_THRESH = 1e-5   # 3e-4→1e-5: 무음 필터 완화

    def pick_input_device(name_hints=NAME_HINTS) -> Optional[int]:
        try:
            devs = sd.query_devices()
        except Exception as e:
            print(f"[audio] query_devices error: {e}")
            return None
        # 힌트 우선
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                nm = d.get("name", "")
                if any(h in nm.upper() for h in name_hints):
                    return i
        # 아니면 첫 입력
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                return i
        return None

    INPUT_INDEX = pick_input_device()
    # 진단 로그
    try:
        devs = sd.query_devices()
        chosen_name = devs[INPUT_INDEX]["name"] if INPUT_INDEX is not None else "(default)"
        print(f"[audio] HAVE_SMILE=True, device_index={INPUT_INDEX}, name={chosen_name}")
    except Exception:
        print(f"[audio] HAVE_SMILE=True, device_index={INPUT_INDEX}")

    sd.default.channels = CHANNELS
    sd.default.samplerate = SAMPLE_RATE
    if INPUT_INDEX is not None:
        sd.default.device = (INPUT_INDEX, None)

    class RealTimeOpenSmile:
        def __init__(self, device_index=INPUT_INDEX, samplerate=SAMPLE_RATE,
                     buffer_seconds=BUFFER_SECONDS, frame_hz=FRAME_HZ, on_features=None):
            self.device = device_index
            self.samplerate = int(samplerate)
            self.buffer_samples = int(self.samplerate * buffer_seconds)
            from collections import deque
            self.buffer = deque(maxlen=self.buffer_samples)
            self.lock = threading.Lock()
            self.frame_interval = 1.0 / float(frame_hz)
            self.on_features = on_features

            self.smile = pyopensmile.Smile(
                feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
                feature_level=pyopensmile.FeatureLevel.Functionals,
            )
            feature_map = {
                "loudness": "loudness_sma3_amean",
                "pitch":    "F0semitoneFrom27.5Hz_sma3nz_amean",
                "jitter":   "jitterLocal_sma3nz_amean",
                "shimmer":  "shimmerLocaldB_sma3nz_amean",
                "hnr":      "HNRdBACF_sma3nz_amean",
            }
            available = set(self.smile.feature_names)
            self.features = {n: k for n, k in feature_map.items() if k in available}

        def _audio_callback(self, indata, frames, time_info, status):
            if status:
                print(f"[audio] sd status: {status}")
            with self.lock:
                self.buffer.extend(indata[:, 0])

        def _process_buffer(self):
            with self.lock:
                if len(self.buffer) < self.buffer_samples:
                    return
                audio = np.array(self.buffer, dtype=np.float32)

            rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
            # 무음이어도 0.0으로 메시지 push (동시성 항상 확인 가능)
            if rms < SILENCE_RMS_THRESH:
                if callable(self.on_features):
                    self.on_features({})  # 빈 특징 → on_features에서 0.0 확률로 전송
                return

            target = 0.05
            gain = min(20.0, target / max(rms, 1e-8))
            audio = np.clip(audio * gain, -1.0, 1.0)

            feats = self.smile.process_signal(audio, self.samplerate)
            series = feats.iloc[0]
            values = {name: float(series.get(key, float("nan"))) for name, key in self.features.items()}

            if callable(self.on_features):
                self.on_features(values)

        def run(self):
            try:
                sd.check_input_settings(
                    device=self.device if self.device is not None else sd.default.device[0],
                    channels=CHANNELS, samplerate=self.samplerate, dtype="float32",
                )
            except Exception as e:
                print("[audio] warn(check_input_settings):", e)

            stream_kwargs = dict(
                samplerate=self.samplerate, channels=CHANNELS, blocksize=BLOCKSIZE,
                dtype="float32", callback=self._audio_callback, latency='high',
            )
            if self.device is not None:
                stream_kwargs["device"] = self.device

            try:
                with sd.InputStream(**stream_kwargs):
                    # 워밍업 (buffer_seconds * 2 초까지 대기)
                    t0 = time.time()
                    warmup_timeout = (self.buffer_samples / self.samplerate) * 2.0
                    while True:
                        with self.lock:
                            filled = len(self.buffer)
                        if filled >= self.buffer_samples or (time.time() - t0) > warmup_timeout:
                            break
                        time.sleep(0.05)

                    next_time = time.monotonic()
                    while not EV_STOP.is_set():
                        now = time.monotonic()
                        if now >= next_time:
                            self._process_buffer()
                            next_time += self.frame_interval
                        time.sleep(0.001)
            except Exception as e:
                print(f"[audio] ERROR opening/reading stream: {e}")

    # openSMILE 특징 → surprise 확률 (간단 휴리스틱)
    def _sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
    def audio_surprise_from_features(values: Dict[str, float]) -> float:
        if not hasattr(audio_surprise_from_features, "_state"):
            audio_surprise_from_features._state = {
                "loud_hist": [], "pitch_hist": [], "prev_pitch": None, "maxlen": 60
            }
        st = audio_surprise_from_features._state
        maxlen = st["maxlen"]

        loud = values.get("loudness", float("nan"))
        pitch = values.get("pitch", float("nan"))

        if isinstance(loud, float) and not math.isnan(loud):
            st["loud_hist"].append(loud); st["loud_hist"] = st["loud_hist"][-maxlen:]
        dpitch = 0.0
        if isinstance(pitch, float) and not math.isnan(pitch):
            if st["prev_pitch"] is not None and isinstance(st["prev_pitch"], float):
                dpitch = abs(pitch - st["prev_pitch"])
            st["prev_pitch"] = pitch
            st["pitch_hist"].append(dpitch); st["pitch_hist"] = st["pitch_hist"][-maxlen:]

        def zscore(arr, x):
            if len(arr) < 10: return 0.0
            m = float(sum(arr)/len(arr))
            s = (sum((a-m)**2 for a in arr)/len(arr))**0.5 or 1e-6
            return max(0.0, (x - m) / s)

        z_loud = zscore(st["loud_hist"], loud if not math.isnan(loud) else 0.0)
        z_dpit = zscore(st["pitch_hist"], dpitch)

        score = 0.8*z_loud + 0.6*z_dpit
        return float(_sigmoid(0.8*(score - 1.5)))

    def audio_producer_opensmile():
        seq = 0
        def on_features(values: Dict[str, float]):
            nonlocal seq
            # 빈 dict(무음) → 확률 0.0으로 전송
            prob = audio_surprise_from_features(values) if values else 0.0
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": prob}, src="opensmile", seq=seq))
            seq += 1

        print("[audio] starting opensmile producer...")
        analyzer = RealTimeOpenSmile(
            device_index=INPUT_INDEX,
            samplerate=SAMPLE_RATE,
            buffer_seconds=BUFFER_SECONDS,
            frame_hz=FRAME_HZ,
            on_features=on_features,
        )
        analyzer.run()
        print("[audio] analyzer.run() exited")

else:
    # 폴백: 랜덤 오디오
    def audio_producer_opensmile():
        period = 1.0 / CFG.audio_hz
        seq = 0
        print("[audio] fallback(random) producer running...")
        while not EV_STOP.is_set():
            val = max(0.0, min(1.0, random.gauss(0.15, 0.1)))
            if random.random() < 0.02:
                val = 0.7 + random.random()*0.3
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": val}, src="audio_fallback", seq=seq))
            seq += 1
            time.sleep(period)

# ---- SERIAL (실차/가상) ----
def serial_producer_real(port="/dev/ttyUSB0", baud=115200):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    S_val = R_val = P_val = B_val = None
    seq = 0
    while not EV_STOP.is_set():
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                try: ser.reset_input_buffer()
                except Exception: pass
                while not EV_STOP.is_set():
                    raw = ser.readline()
                    if not raw: continue
                    line = raw.decode('utf-8', errors='ignore').strip()
                    m = pattern.fullmatch(line)
                    if not m: continue
                    new_S = int(m.group(1)); new_R = int(m.group(2))
                    new_P = float(m.group(3)); new_B = int(m.group(4))
                    if not (S_val is not None and S_val > 0 and new_S == 0):
                        S_val = new_S
                    if new_R != 0: R_val = new_R
                    if new_P != 0.0: P_val = new_P
                    B_val = new_B
                    if None in (S_val, R_val, P_val, B_val): continue
                    data = {"speed": float(S_val), "rpm": int(R_val),
                            "throttle": float(P_val), "brake": int(B_val)}
                    Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_serial", seq=seq))
                    seq += 1
        except Exception as e:
            print(f"[serial] reconnecting due to: {e}")
            time.sleep(1.0)

def serial_producer_loopback_sim(rate_hz: float = 50.0):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    ser = (_serial_for_url("loop://", baudrate=115200, timeout=1)
           if _serial_for_url else serial.serial_for_url("loop://", baudrate=115200, timeout=1))
    dt = 1.0 / rate_hz
    def writer():
        while not EV_STOP.is_set():
            for _ in range(int(3.0 * rate_hz)):
                ser.write(b"/S30/R2000/P0.20/B0\n"); time.sleep(dt)
            for _ in range(int(0.5 * rate_hz)):
                ser.write(b"/S10/R3500/P0.90/B1\n"); time.sleep(dt)
            for _ in range(int(10.0 * rate_hz)):     # 10초 지속: 오디오 워밍업 이후도 관찰
                ser.write(b"/S5/R4000/P0.90/B1\n"); time.sleep(dt)
            for _ in range(int(3.0 * rate_hz)):
                ser.write(b"/S30/R2000/P0.20/B0\n"); time.sleep(dt)
    threading.Thread(target=writer, daemon=True).start()

    S_val = R_val = P_val = B_val = None
    seq = 0
    while not EV_STOP.is_set():
        raw = ser.readline()
        if not raw: continue
        line = raw.decode("utf-8", errors="ignore").strip()
        m = pattern.fullmatch(line)
        if not m: continue
        new_S = int(m.group(1)); new_R = int(m.group(2))
        new_P = float(m.group(3)); new_B = int(m.group(4))
        if not (S_val is not None and S_val > 0 and new_S == 0):
            S_val = new_S
        if new_R != 0: R_val = new_R
        if new_P != 0.0: P_val = new_P
        B_val = new_B
        if None in (S_val, R_val, P_val, B_val): continue
        data = {"speed": float(S_val), "rpm": int(R_val),
                "throttle": float(P_val), "brake": int(B_val)}
        Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_loop", seq=seq))
        seq += 1

# ==========================
# 4) Consumer(판단 루프)
# ==========================

@dataclass
class FusionState:
    horizon: float
    audio: List[Msg] = field(default_factory=list)
    video: List[Msg] = field(default_factory=list)
    serial: List[Msg] = field(default_factory=list)

    def add(self, m: Msg):
        if m.type == MsgType.AUDIO:
            self.audio.append(m)
        elif m.type == MsgType.VIDEO:
            self.video.append(m)
        elif m.type == MsgType.SERIAL:
            self.serial.append(m)
        prune_buffer(self.audio, self.horizon)
        prune_buffer(self.video, self.horizon)
        prune_buffer(self.serial, self.horizon)

def is_obd_abnormal(state: FusionState) -> (bool, List[str]):
    why = []
    if not state.serial:
        return False, why
    cur = state.serial[-1].data
    brake = int(cur.get("brake", 0))
    throttle = float(cur.get("throttle", 0.0))
    speed = float(cur.get("speed", 0.0))
    drpm = delta_over_window(state.serial, "rpm", CFG.recent_window_sec)
    dspeed = delta_over_window(state.serial, "speed", CFG.recent_window_sec)
    cond1 = (brake == 1 and throttle > CFG.throttle_brake_conflict_thr)
    cond2 = (throttle > CFG.throttle_high_thr and speed < CFG.low_speed_thr)
    cond4 = (drpm > CFG.rpm_jump_thr and abs(dspeed) < CFG.speed_small_thr)
    if cond1: why.append("OBD:brake+throttle_conflict")
    if cond2: why.append("OBD:high_throttle_low_speed")
    if cond4: why.append("OBD:rpm_jump_speed_static")
    return (cond1 or cond2 or cond4), why

def decide(state: FusionState, last_alert_ts: float) -> Dict[str, Any]:
    decision = "OK"
    reasons: List[str] = []
    score = 0.0

    obd_abn, obd_why = is_obd_abnormal(state)
    if obd_abn:
        reasons.extend(obd_why); score += 1.0

    A_peak = peak(state.audio, "audio_surprise_prob", CFG.recent_window_sec)
    V_peak = peak(state.video, "face_surprise_prob", CFG.recent_window_sec)
    A_persist = count_over(state.audio, "audio_surprise_prob", CFG.TA, CFG.recent_window_sec)
    V_persist = count_over(state.video, "face_surprise_prob", CFG.TV, CFG.recent_window_sec)
    human_peak = max(A_peak, V_peak)

    if obd_abn and (A_persist >= CFG.persist_min_frames or V_persist >= CFG.persist_min_frames):
        decision = "ALERT"; reasons.append("HUMAN:persistent"); score += 0.5
    elif obd_abn and human_peak > max(CFG.TA, CFG.TV) + 0.1:
        decision = "WARNING"; reasons.append("HUMAN:peak"); score += 0.3
    else:
        decision = "OK"

    decision_ts = now_s()
    if decision == "ALERT" and (decision_ts - last_alert_ts) < CFG.cooldown_sec:
        decision = "OK"; reasons.append("cooldown_suppressed")

    cur_obd = state.serial[-1].data if state.serial else {}
    last_audio = state.audio[-1] if state.audio else None
    last_video = state.video[-1] if state.video else None
    last_serial = state.serial[-1] if state.serial else None

    def _lag_ms(m): return None if m is None else round((decision_ts - m.ts) * 1000.0, 1)
    def _count_recent(buf, sec=1.0): t0 = decision_ts - sec; return sum(1 for m in buf if m.ts >= t0)

    result = {
        "ts": decision_ts,
        "decision": decision,
        "score": round(score, 3),
        "why": reasons,
        "obd_snapshot": {
            "speed": cur_obd.get("speed"),
            "throttle": cur_obd.get("throttle"),
            "brake": cur_obd.get("brake"),
            "rpm": cur_obd.get("rpm"),
        },
        "peaks": {"audio": round(A_peak, 3), "video": round(V_peak, 3)},
        "persist": {"audio": A_persist, "video": V_persist},
        "window": {"len_sec": CFG.buffer_sec, "recent_sec": CFG.recent_window_sec},

        # 동시성/진단
        "sources_ts": {
            "audio": (last_audio.ts if last_audio else None),
            "video": (last_video.ts if last_video else None),
            "serial": (last_serial.ts if last_serial else None),
        },
        "sources_lag_ms": {
            "audio": _lag_ms(last_audio),
            "video": _lag_ms(last_video),
            "serial": _lag_ms(last_serial),
        },
        "last_seq": {
            "audio": (last_audio.seq if last_audio else None),
            "video": (last_video.seq if last_video else None),
            "serial": (last_serial.seq if last_serial else None),
        },
        "sources_count_recent_1s": {
            "audio": _count_recent(state.audio, 1.0),
            "video": _count_recent(state.video, 1.0),
            "serial": _count_recent(state.serial, 1.0),
        },
        "sources_src": {
            "audio": (last_audio.src if last_audio else None),
            "video": (last_video.src if last_video else None),
            "serial": (last_serial.src if last_serial else None),
        }
    }
    return result

# ==================
# 5) 로깅/출력 유틸
# ==================

def emit_event(ev: Dict[str, Any]):
    line = json.dumps(ev, ensure_ascii=False)
    print(line)
    if CFG.log_to_file:
        with open(CFG.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# =====================
# 6) 메인 루프/런처
# =====================

def decision_loop():
    state = FusionState(horizon=CFG.buffer_sec)
    last_emit = 0.0
    last_alert_ts = -1e9
    while not EV_STOP.is_set():
        try:
            m: Msg = Q.get(timeout=0.1)
        except queue.Empty:
            continue
        state.add(m)
        now = now_s()
        if (now - last_emit) >= 0.05:  # ~20Hz
            res = decide(state, last_alert_ts)
            if res["decision"] in ("WARNING", "ALERT"):
                if res["decision"] == "ALERT":
                    last_alert_ts = res["ts"]
                emit_event(res)
            last_emit = now

def main():
    ths = [
        threading.Thread(target=audio_producer_opensmile, daemon=True),
        threading.Thread(target=video_producer, daemon=True),
        threading.Thread(target=serial_producer_loopback_sim, kwargs={"rate_hz": 50.0}, daemon=True),
        # 실제 차량 시리얼 사용 시 ↑ 대신 ↓
        # threading.Thread(target=serial_producer_real, kwargs={"port": "/dev/ttyUSB0", "baud": 115200}, daemon=True),
        threading.Thread(target=decision_loop, daemon=True),
    ]
    for t in ths: t.start()
    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        EV_STOP.set()
        for t in ths:
            t.join(timeout=1.0)
        print("Stopped.")

if __name__ == "__main__":
    main()
