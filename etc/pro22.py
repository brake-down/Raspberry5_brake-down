#!/usr/bin/env python3
"""
Real-time pedal misoperation detector (fusion of VIDEO/AUDIO/OBD2)
- 입력: VIDEO 1개 확률값, AUDIO(마이크→openSMILE) 1개 확률값, OBD2(시리얼) 4개 값 {speed,rpm,throttle,brake}
- 구조: Producer(3) -> Queue -> Consumer(판단 루프) -> 경고/로그
"""

import time
import json
import queue
import threading
import re
import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

from collections import deque
import traceback 

# ========= [AUDIO deps - lazy import] =========
_HAS_AUDIO_STACK = True
try:
    import sounddevice as sd
except Exception as _e:
    _HAS_AUDIO_STACK = False
    sd = None

try:
    # 사용자가 셔틀 파일(pyopensmile.py)을 둘 수도 있으므로 우선 시도
    import pyopensmile as pyopensmile
except Exception:
    try:
        import opensmile as pyopensmile
    except Exception:
        _HAS_AUDIO_STACK = False
        pyopensmile = None

# ========= [SERIAL deps] =========
import serial
from serial import serial_for_url

# =========================
# 0) 설정(튜닝 가능한 값들)
# =========================

@dataclass
class Config:
    # 버퍼/창
    buffer_sec: float = 2.0          # 최근 2~3초 권장, 초기 2.0s
    recent_window_sec: float = 0.5   # 최근 0.5s 창에서 피크/지속성 계산
    # 임계값
    TA: float = 0.5                  # audio_surprise 임계
    TV: float = 0.5                  # face_surprise 임계
    persist_min_frames: int = 3      # 최근 0.5s 창에서 임계 이상 프레임 수
    # OBD 비정상 판단 임계
    throttle_brake_conflict_thr: float = 0.3   # brake=1 & throttle>0.3
    throttle_high_thr: float = 0.8             # throttle>0.8 & speed<2
    low_speed_thr: float = 2.0
    rpm_jump_thr: float = 1200.0               # 최근 0.5s 내 rpm 급증
    speed_small_thr: float = 1.0               # 최근 0.5s 내 speed 거의 무변화
    # 쿨다운
    cooldown_sec: float = 2.0
    # 주기(모의/비디오)
    audio_hz: float = 20.0
    video_hz: float = 20.0
    obd_hz: float = 50.0
    # 로깅
    log_to_file: bool = False
    log_path: str = "events.log"

    # ===== (NEW) AUDIO 실시간 분석 파이프 설정 =====
    # 라즈베리파이5 + RØDE 기반 기본값. 필요시 조절 가능.
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_blocksize: int = 4096
    audio_frame_hz: float = 20.0        # 큐에 넣는 빈도(결정 루프와 비슷하게)
    audio_buffer_seconds: float = 2.0    # 슬라이딩 윈도 길이(짧을수록 반응 빠름)
    audio_silence_rms_thresh: float = 3e-4  # 무음 게이트 (NaN 방지/잡음 억제)
    audio_gain_target_rms: float = 0.05     # 자동 게인 타겟

    audio_try_samplerates: List[int] = field(default_factory=lambda: [44100])
    audio_try_channels:   List[int] = field(default_factory=lambda: [1, 2])
    
    audio_open_timeout_sec: float = 1.5   # 스트림 오픈/콜백 대기 타임아웃
    audio_cb_dead_after_sec: float = 1.5  # 콜백이 n초 동안 0프레임이면 다음 조합으로 폴백
    audio_print_devices: bool = True      # 시작 시 장치 테이블 덤프

    # (NEW) openSMILE 특징 → surprise 확률 맵핑 파라미터(대략적 경험값)
    feat_norms: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # min/max는 경험적 스케일링용, 필요시 튜닝
        "loudness": {"min": 0.0, "max": 1.0, "w": 0.40},
        "jitter":   {"min": 0.0, "max": 0.5, "w": 0.20},
        "shimmer":  {"min": 0.0, "max": 6.0, "w": 0.20},
        "hnr_inv":  {"min": 0.0, "max": 20.0, "w": 0.10},  # hnr가 낮을수록 놀람↑ → inverse
        "dpitch":   {"min": 0.0, "max": 2.0, "w": 0.10},   # pitch의 변화량(세미톤) 정규화
    })
    audio_logit_bias: float = -1.2  # 로지스틱 바이어스(기본 낮게)
    
    audio_debug_print: bool = True

CFG = Config()

# ======================
# 1) 공통 메시지/큐 정의
# ======================

class MsgType(Enum):
    VIDEO = auto()
    AUDIO = auto()
    SERIAL = auto()  # OBD2

@dataclass
class Msg:
    type: MsgType
    ts: float
    data: Dict[str, Any]
    src: str = ""                 # optional: 장치/모듈명
    seq: Optional[int] = None     # optional: 시퀀스 번호
    meta: Dict[str, Any] = field(default_factory=dict)

# 전역 큐/종료 이벤트
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
# 3) Producer(입력)
# ===========================

# ---- (A) AUDIO: openSMILE 기반 실시간 Producer ----

def _pick_input_device(name_hints=("RODE", "RØDE", "VIDEOMIC")) -> Optional[int]:
    """RØDE 계열 우선 선택, 없으면 첫 입력 장치."""
    if not sd:
        return None
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                nm = (d.get("name", "") or "").upper()
                if any(h in nm for h in name_hints):
                    return i
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                return i
    except Exception as e:
        print(f"[audio] device query failed: {e}", file=sys.stderr)
    return None

def _dump_devices():
    try:
        devs = sd.query_devices()
        print("[audio] ==== PortAudio devices ====")
        for i, d in enumerate(devs):
            print(f"[{i}] {d.get('name')}  in:{d.get('max_input_channels')}  out:{d.get('max_output_channels')}  sr:{int(d.get('default_samplerate', 0))}")
        print("[audio] ===========================")
    except Exception as e:
        print(f"[audio] device list failed: {e}")

def _input_device_candidates():
    """입력 채널이 있는 디바이스 인덱스만 반환(우선순위: pulse/default 먼저)."""
    try:
        devs = sd.query_devices()
    except Exception as e:
        print(f"[audio] device query failed in candidates: {e}")
        return [None]
    
    idxs = []
    # pulse/default 먼저
    for i, d in enumerate(devs):
        name = (d.get("name") or "").lower()
        if d.get("max_input_channels", 0) > 0 and ("pulse" in name or "default" in name):
            idxs.append(i)
    # 나머지 입력 가능한 장치
    for i, d in enumerate(devs):
        name = (d.get("name") or "").lower()
        if d.get("max_input_channels", 0) > 0 and not ("pulse" in name or "default" in name):
            idxs.append(i)
    # 자동선택(None)도 후보 맨 뒤에 추가
    idxs.append(None)
    # 중복 제거
    seen, ordered = set(), []
    for x in idxs:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered

def _device_default_sr(idx):
    """해당 디바이스의 default samplerate를 반환. idx가 None이면 44100을 반환."""
    try:
        if idx is None:
            return 44100
        info = sd.query_devices(idx)
        sr = int(info.get("default_samplerate", 44100) or 44100)
        return sr
    except Exception:
        return 44100

def _can_open_input(dev, sr, ch) -> (bool, str):
    try:
        sd.check_input_settings(device=dev, channels=ch, samplerate=sr, dtype="float32")
        return True, ""
    except Exception as e:
        return False, str(e)

def _open_stream_with_timeout(kwargs, timeout_sec=1.2):
    result = {"stream": None, "err": None}
    done = threading.Event()

    def worker():
        try:
            s = sd.InputStream(**kwargs)
            s.start()
            result["stream"] = s
        except Exception as e:
            result["err"] = e
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ok = done.wait(timeout_sec)
    if not ok:
        return None, TimeoutError("open/start timeout")
    if result["stream"] is None:
        return None, result["err"]
    return result["stream"], None


def _can_open_input(dev, sr, ch) -> (bool, str):
    try:
        sd.check_input_settings(device=dev, channels=ch, samplerate=sr, dtype="float32")
        return True, ""
    except Exception as e:
        return False, str(e)

def _open_stream_with_timeout(kwargs, timeout_sec=1.2):
    """
    sd.InputStream(**kwargs) + start() 를 별도 스레드에서 돌리고
    timeout 안에 안 끝나면 None 반환(=블록 회피).
    """
    result = {"stream": None, "err": None}
    done = threading.Event()

    def worker():
        try:
            s = sd.InputStream(**kwargs)
            s.start()
            result["stream"] = s
        except Exception as e:
            result["err"] = e
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ok = done.wait(timeout_sec)
    if not ok:
        result["err"] = TimeoutError("open/start timeout")
        return None, result["err"]
    if result["stream"] is None:
        return None, result["err"]
    return result["stream"], None


class AudioProducerOpenSmile:
    """
    - 사운드 입력을 슬라이딩 버퍼에 모아 일정 주기로 openSMILE GeMAPS(Functionals) 특징 계산
    - 특징을 간단한 휴리스틱으로 [0..1] audio_surprise_prob로 변환하여 큐에 투입
    """
    def __init__(self):
        self.enabled = _HAS_AUDIO_STACK
        self.device = _pick_input_device()
        self.sr = CFG.audio_sample_rate
        self.ch = CFG.audio_channels
        self.blocksize = CFG.audio_blocksize
        self.frame_interval = 1.0 / CFG.audio_frame_hz
        self.buffer_samples = int(self.sr * CFG.audio_buffer_seconds)
        #self.buf: "queue.deque[float]" = queue.deque(maxlen=self.buffer_samples)
        self.buf: deque[float] = deque(maxlen=self.buffer_samples)

        self.lock = threading.Lock()
        self.seq = 0
        
        self._last_cb = 0.0   # 마지막 콜백 도착시각(헬스체크용)


        if not self.enabled:
            print("[audio] openSMILE/SD 미탑재 → 모의 producer로 폴백")
            return

        try:
            # openSMILE 구성: GeMAPS(Functionals)
            self.smile = pyopensmile.Smile(
                feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
                feature_level=pyopensmile.FeatureLevel.Functionals,
            )
        except Exception as e:
            print(f"[audio] openSMILE init 실패: {e} → 모의 producer로 폴백")
            self.enabled = False
            return
        
        print(f"[audio] device={self.device}, sr={self.sr}, ch={self.ch}, blocksize={self.blocksize}")


        # feature 키 매핑
        fmap = {
            "loudness": "loudness_sma3_amean",
            "pitch":    "F0semitoneFrom27.5Hz_sma3nz_amean",
            "jitter":   "jitterLocal_sma3nz_amean",
            "shimmer":  "shimmerLocaldB_sma3nz_amean",
            "hnr":      "HNRdBACF_sma3nz_amean",
        }
        available = set(self.smile.feature_names)
        self.fkeys = {n: k for n, k in fmap.items() if k in available}
        missing = set(fmap) - set(self.fkeys)
        if missing:
            print(f"[audio] 일부 특징 미제공: {missing} (계산엔 영향 없음)")

        # 피치 EMA(변화량 계산용)
        self.pitch_ema = None
        self.ema_alpha = 0.25  # EMA 완만도

        # sounddevice 기본값 지정
        try:
            sd.default.channels = self.ch
            sd.default.samplerate = self.sr
            #if self.device is not None:
            #    sd.default.device = (self.device, None)
            # 사전 설정 검증
            sd.check_input_settings(
                device=self.device if self.device is not None else sd.default.device[0],
                channels=self.ch,
                samplerate=self.sr,
                dtype="float32",
            )
        except Exception as e:
            print(f"[audio] 장치 설정 경고: {e}")
            
    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # 채널 수가 1 또는 2 등 무엇이든, 항상 모노로 변환해 버퍼에 추가
        try:
            if indata.ndim == 2 and indata.shape[1] > 1:
                mono = np.mean(indata, axis=1, dtype=np.float32)
            else:
                mono = indata[:, 0] if indata.ndim == 2 else indata.astype(np.float32)
        except Exception as e:
            print(f"[audio] cb shape err: {indata.shape if hasattr(indata,'shape') else None} -> {e}")
            return
        with self.lock:
            self.buf.extend(mono)
        self._last_cb = time.time()  # ★ 콜백 헬스비트 기록
        # 1초에 1회 정도 콜백 살아있는지 출력
        if getattr(self, "_cb_log_t", 0) + 1.0 < self._last_cb:
            self._cb_log_t = self._last_cb
            print(f"[audio] cb ok (frames={frames}, buf={len(self.buf)}/{self.buffer_samples})")

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _normalize(self, v: float, vmin: float, vmax: float, invert=False) -> float:
        if invert:
            v = vmax - (v - vmin)
        denom = max(1e-6, (vmax - vmin))
        return float(np.clip((v - vmin) / denom, 0.0, 1.0))

    def _features_to_prob(self, feats: Dict[str, float]) -> float:
        """
        loudness, jitter, shimmer, hnr(역), pitch 변화량을 가중합 → 로지스틱
        """
        # dpitch 계산(세미톤 기준 EMA 대비 변화량)
        pitch = float(feats.get("pitch", 0.0))
        if self.pitch_ema is None:
            self.pitch_ema = pitch
        self.pitch_ema = (1.0 - self.ema_alpha) * self.pitch_ema + self.ema_alpha * pitch
        dpitch = abs(pitch - self.pitch_ema)

        # 정규화
        fcfg = CFG.feat_norms
        x_loud = self._normalize(feats.get("loudness", 0.0), fcfg["loudness"]["min"], fcfg["loudness"]["max"])
        x_jit  = self._normalize(feats.get("jitter",   0.0), fcfg["jitter"]["min"],   fcfg["jitter"]["max"])
        x_shi  = self._normalize(feats.get("shimmer",  0.0), fcfg["shimmer"]["min"],  fcfg["shimmer"]["max"])
        # hnr 낮을수록 놀람↑ → invert 효과를 위해 (max - value) 정규화처럼 취급
        # 여기선 단순히 (max - hnr)를 정규화하는 대신, hnr_inv = max(0, target - hnr) 같은 느낌으로 사용
        hnr = feats.get("hnr", 0.0)
        x_hinv = self._normalize(max(0.0, 20.0 - hnr), fcfg["hnr_inv"]["min"], fcfg["hnr_inv"]["max"])
        x_dpi  = self._normalize(dpitch, fcfg["dpitch"]["min"], fcfg["dpitch"]["max"])

        # 가중합 → logit
        s = (
            fcfg["loudness"]["w"] * x_loud +
            fcfg["jitter"]["w"]   * x_jit  +
            fcfg["shimmer"]["w"]  * x_shi  +
            fcfg["hnr_inv"]["w"]  * x_hinv +
            fcfg["dpitch"]["w"]   * x_dpi  +
            CFG.audio_logit_bias
        )
        return float(self._sigmoid(s))

    def run(self):
        if not self.enabled:
            print("[audio] fallback random producer running...")
            _mock_audio_producer()
            return

        # 입력 가능한 장치 후보 획득(pulse/default 우선)
        device_candidates = _input_device_candidates()
        # default(2)는 제외. pulse(1)를 최우선으로.
        device_candidates = [d for d in device_candidates if d != 2]
        if 1 in device_candidates:
            device_candidates = [1] + [d for d in device_candidates if d != 1]
        print(f"[audio] device candidates (input-capable incl. None): {device_candidates}")

        tried = []
        for dev in device_candidates:
            # 해당 디바이스의 기본 샘플레이트를 최우선 시도
            dev_sr = _device_default_sr(dev)
            # Pulse 소스가 48k/1ch 이므로 우선
            sr_order = [48000, 44100]
            ch_order = [1, 2]
            for sr in sr_order:
                for ch in ch_order:
                    print(f"[audio] trying device={dev}, sr={sr}, ch={ch}, blocksize={self.blocksize}")
                    self.sr = sr
                    self.ch = ch
                    self.buffer_samples = int(self.sr * CFG.audio_buffer_seconds)
                    self.buf = deque(maxlen=self.buffer_samples)
                    self._last_cb = 0.0

                    kwargs = dict(
                        samplerate=sr, channels=ch, blocksize=0,
                        dtype="float32", callback=self._audio_cb, latency="high"
                    )
                    # 1) 사전체크 (dev가 None이면 건너뜀)
                    if dev is not None:
                        ok, emsg = _can_open_input(dev, sr, ch)
                        if not ok:
                            print(f"[audio] precheck skip device={dev}, sr={sr}, ch={ch} -> {emsg}")
                            continue

                    stream = None
                    opened = False
                    try:
                        stream, err = _open_stream_with_timeout(kwargs, timeout_sec=CFG.audio_open_timeout_sec)
                        if stream is None:
                            raise RuntimeError(str(err) if err else "open failed")
                        opened = True
                        print("[audio] stream opened & started")

                        # 콜백 대기
                        t0 = time.time()
                        while (time.time() - t0) < CFG.audio_open_timeout_sec:
                            if self._last_cb > 0:
                                break
                            time.sleep(0.05)
                        if self._last_cb == 0:
                            raise RuntimeError("no audio callback within open-timeout")
                        print("[audio] first callback received")

                        # 워밍업
                        while True:
                            with self.lock:
                                filled = len(self.buf)
                            if filled >= self.buffer_samples:
                                print("[audio] warmup done")
                                break
                            if (time.time() - self._last_cb) > CFG.audio_cb_dead_after_sec:
                                raise RuntimeError("audio callback stalled during warmup")
                            time.sleep(0.05)

                        # 주기 처리
                        next_t = time.monotonic()
                        while not EV_STOP.is_set():
                            now = time.monotonic()
                            if now >= next_t:
                                self._process_once()
                                next_t += self.frame_interval
                            if (time.time() - self._last_cb) > CFG.audio_cb_dead_after_sec:
                                raise RuntimeError("audio callback stalled")
                            time.sleep(0.001)

                        stream.stop(); stream.close()
                        return

                    except Exception as e:
                        tried.append((dev, sr, ch, str(e)))
                        print(f"[audio] open/run failed device={dev}, sr={sr}, ch={ch}: {e}")
                        try:
                            if opened and stream:
                                stream.stop(); stream.close()
                        except Exception:
                            pass
                        continue


                        # 오픈 후 콜백 타임아웃 대기
                        t0 = time.time()
                        while (time.time() - t0) < CFG.audio_open_timeout_sec:
                            if self._last_cb > 0:
                                break
                            time.sleep(0.05)
                        if self._last_cb == 0:
                            raise RuntimeError("no audio callback within open-timeout")
                        print("[audio] first callback received")

                        # 워밍업(버퍼 채우기)
                        while True:
                            with self.lock:
                                filled = len(self.buf)
                            if filled >= self.buffer_samples:
                                print("[audio] warmup done")
                                break
                            if (time.time() - self._last_cb) > CFG.audio_cb_dead_after_sec:
                                raise RuntimeError("audio callback stalled during warmup")
                            time.sleep(0.05)

                        # 주기 처리
                        next_t = time.monotonic()
                        while not EV_STOP.is_set():
                            now = time.monotonic()
                            if now >= next_t:
                                self._process_once()
                                next_t += self.frame_interval
                            if (time.time() - self._last_cb) > CFG.audio_cb_dead_after_sec:
                                raise RuntimeError("audio callback stalled")
                            time.sleep(0.001)

                        stream.stop(); stream.close()
                        return

                    except Exception as e:
                        tried.append((dev, sr, ch, str(e)))
                        print(f"[audio] open/run failed device={dev}, sr={sr}, ch={ch}: {e}")
                        try:
                            if opened and stream:
                                stream.stop(); stream.close()
                        except Exception:
                            pass
                        continue

        print("[audio] all attempts failed; tried:", tried)
        _mock_audio_producer()





    def _process_once(self):
        # 버퍼에서 최신 오디오 스냅샷 취득
        with self.lock:
            if len(self.buf) < self.buffer_samples:
                return
            audio = np.array(self.buf, dtype=np.float32)

        # 무음 게이트
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < CFG.audio_silence_rms_thresh:
            if CFG.audio_debug_print:
                print(f"(silence) waiting... rms={rms:.6g}")
            # 무음 구간에선 놀람 확률 매우 낮게
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.05}, src="audio_os", seq=self.seq))
            self.seq += 1
            return

        # 자동게인 → openSMILE 입력
        gain = min(20.0, CFG.audio_gain_target_rms / max(rms, 1e-8))
        audio = np.clip(audio * gain, -1.0, 1.0)

        try:
            feats_df = self.smile.process_signal(audio, self.sr)
        except Exception as e:
            print(f"[audio] openSMILE process 실패: {e}")
            return

        s = feats_df.iloc[0]
        feats = {
            "loudness": float(s.get(self.fkeys.get("loudness", ""), np.nan)),
            "pitch":    float(s.get(self.fkeys.get("pitch",    ""), np.nan)),
            "jitter":   float(s.get(self.fkeys.get("jitter",   ""), np.nan)),
            "shimmer":  float(s.get(self.fkeys.get("shimmer",  ""), np.nan)),
            "hnr":      float(s.get(self.fkeys.get("hnr",      ""), np.nan)),
        }

        # NaN 방지: 결측은 0으로
        for k, v in list(feats.items()):
            if not np.isfinite(v):
                feats[k] = 0.0
                
        if CFG.audio_debug_print:
            print(
                "loudness: {0:.2f} | pitch: {1:.2f} | jitter: {2:.2f} | shimmer: {3:.2f} | hnr: {4:.2f}"
                .format(feats["loudness"], feats["pitch"], feats["jitter"], feats["shimmer"], feats["hnr"])
            )

        prob = self._features_to_prob(feats)
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": float(np.clip(prob, 0.0, 1.0))},
                  src="audio_os", seq=self.seq, meta={"feats": feats}))
        self.seq += 1
    




def _mock_audio_producer():
    """openSMILE/장치 불가 시 사용하는 랜덤 폴백(이전 코드의 모의 입력과 유사)"""
    period = 1.0 / CFG.audio_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        val = max(0.0, min(1.0, random.gauss(0.15, 0.1)))
        if random.random() < 0.02:
            val = 0.7 + random.random()*0.3
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": val}, src="audio_mock", seq=seq))
        seq += 1
        time.sleep(period)

def audio_producer():
    try:
        if CFG.audio_print_devices:
                _dump_devices()
        AudioProducerOpenSmile().run()
    except Exception:
        print("[audio] thread crashed:")
        traceback.print_exc()
        # 안전 폴백: mock 로 전환
        _mock_audio_producer()


# ---- (B) VIDEO: 모의 Producer 유지 ----
def video_producer():
    period = 1.0 / CFG.video_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        val = max(0.0, min(1.0, random.gauss(0.12, 0.1)))
        if random.random() < 0.015:
            val = 0.65 + random.random()*0.35
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": val}, src="cam0", seq=seq))
        seq += 1
        time.sleep(period)

# ---- (C) OBD2: 실제/루프백 Producer 그대로 ----
def serial_producer_real(port="/dev/ttyUSB0", baud=115200):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')

    S_val = R_val = P_val = B_val = None
    while not EV_STOP.is_set():
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                while not EV_STOP.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode('utf-8', errors='ignore').strip()
                    m = pattern.fullmatch(line)
                    if not m:
                        continue

                    new_S = int(m.group(1))
                    new_R = int(m.group(2))
                    new_P = float(m.group(3))
                    new_B = int(m.group(4))

                    if not (S_val is not None and S_val > 0 and new_S == 0):
                        S_val = new_S
                    if new_R != 0:
                        R_val = new_R
                    if new_P != 0.0:
                        P_val = new_P
                    B_val = new_B

                    if None in (S_val, R_val, P_val, B_val):
                        continue

                    data = {"speed": float(S_val), "rpm": int(R_val),
                            "throttle": float(P_val), "brake": int(B_val)}
                    Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_serial"))
        except Exception as e:
            print(f"[serial] reconnecting due to: {e}")
            time.sleep(1.0)

def serial_producer_loopback_sim(rate_hz: float = 50.0):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    ser = serial_for_url("loop://", baudrate=115200, timeout=1)

    def writer():
        dt = 1.0 / rate_hz
        for _ in range(int(3.0 * rate_hz)):
            ser.write(f"/S30/R2000/P0.20/B0\n".encode("utf-8"))
            time.sleep(dt)
        for _ in range(int(0.5 * rate_hz)):
            ser.write(f"/S10/R3500/P0.90/B1\n".encode("utf-8"))
            time.sleep(dt)
        for _ in range(int(2.0 * rate_hz)):
            ser.write(f"/S5/R4000/P0.90/B1\n".encode("utf-8"))
            time.sleep(dt)

    threading.Thread(target=writer, daemon=True).start()

    S_val = R_val = P_val = B_val = None
    while not EV_STOP.is_set():
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        m = pattern.fullmatch(line)
        if not m:
            continue

        new_S = int(m.group(1))
        new_R = int(m.group(2))
        new_P = float(m.group(3))
        new_B = int(m.group(4))

        if not (S_val is not None and S_val > 0 and new_S == 0):
            S_val = new_S
        if new_R != 0:
            R_val = new_R
        if new_P != 0.0:
            P_val = new_P
        B_val = new_B

        if None in (S_val, R_val, P_val, B_val):
            continue

        data = {"speed": float(S_val), "rpm": int(R_val),
                "throttle": float(P_val), "brake": int(B_val)}
        Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_loop"))

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
        reasons.extend(obd_why)
        score += 1.0

    A_peak = peak(state.audio, "audio_surprise_prob", CFG.recent_window_sec)
    V_peak = peak(state.video, "face_surprise_prob", CFG.recent_window_sec)
    A_persist = count_over(state.audio, "audio_surprise_prob", CFG.TA, CFG.recent_window_sec)
    V_persist = count_over(state.video, "face_surprise_prob", CFG.TV, CFG.recent_window_sec)
    human_peak = max(A_peak, V_peak)

    if obd_abn and (A_persist >= CFG.persist_min_frames or V_persist >= CFG.persist_min_frames):
        decision = "ALERT"
        reasons.append("HUMAN:persistent")
        score += 0.5
    elif obd_abn and human_peak > max(CFG.TA, CFG.TV) + 0.1:
        decision = "WARNING"
        reasons.append("HUMAN:peak")
        score += 0.3
    else:
        decision = "OK"

    now = now_s()
    if decision == "ALERT" and (now - last_alert_ts) < CFG.cooldown_sec:
        decision = "OK"
        reasons.append("cooldown_suppressed")

    cur_obd = state.serial[-1].data if state.serial else {}
    result = {
        "ts": now,
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

# ===========================
# 테스트 시뮬레이터 (정상→WARNING→ALERT)
# ===========================

def test_scenario_producer():
    seq = 0
    for _ in range(60):
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.1}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.1}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 0, "throttle": 0.2, "speed": 30, "rpm": 2000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    for _ in range(10):
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.8}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 10, "rpm": 3500
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    for _ in range(40):
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.9}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 5, "rpm": 4000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    EV_STOP.set()

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
        if (now - last_emit) >= 0.05:
            res = decide(state, last_alert_ts)
            if res["decision"] in ("WARNING", "ALERT"):
                if res["decision"] == "ALERT":
                    last_alert_ts = res["ts"]
                emit_event(res)
            last_emit = now

def main():
    ths = [
        threading.Thread(target=audio_producer, daemon=True),           # ★ 마이크 → audio_surprise_prob
        threading.Thread(target=video_producer, daemon=True),

        # 🔁 실제 차량이 없을 땐 loop:// 시뮬레이터
        threading.Thread(target=serial_producer_loopback_sim, kwargs={"rate_hz": 50.0}, daemon=True),

        # ★ 실제 OBD 사용 시 위 줄 주석 처리하고 아래 주석 해제
        # threading.Thread(target=serial_producer_real, kwargs={"port": "/dev/ttyUSB0", "baud": 115200}, daemon=True),

        threading.Thread(target=decision_loop, daemon=True),
    ]

    # === 시나리오만 돌릴 땐 ===
    # ths = [
    #     threading.Thread(target=test_scenario_producer, daemon=True),
    #     threading.Thread(target=decision_loop, daemon=True),
    # ]

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
