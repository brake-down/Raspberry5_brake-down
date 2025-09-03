# gemini

# openSmile1.py
# 실시간 음성 입력을 받아 급발진 확률을 계산하는 스크립트.
# realtime_analyzer.py의 실시간 오디오 처리와 openSmile2.py의 분석 로직을 결합합니다.

import sys
import time
import threading
from collections import deque
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import sounddevice as sd

import math

def num(x, default: float = 0.0) -> float:
    """None/NaN/inf를 안전하게 기본값으로 치환해 float 반환"""
    try:
        if x is None:
            return default
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return default
        return xf
    except Exception:
        return default


# --- pyopensmile 임포트 ---
# openSMILE의 Python 래퍼 라이브러리입니다.
try:
    import pyopensmile
except ImportError:
    try:
        import opensmile as pyopensmile
    except ImportError as e:
        raise RuntimeError(
            "openSMILE python package가 없습니다. `pip install opensmile` 명령으로 설치하세요."
        ) from e

# ======================================================================
# openSmile2.py에서 가져온 분석 로직
# 이 부분은 음성 특징을 기반으로 '급발진/긴장' 상태를 판단하는 규칙과 계산식을 포함합니다.
# ======================================================================

# --- 1. 플래그별 가중치 ---
# 특정 음성 특징(플래그)이 감지되었을 때 확률 계산에 얼마나 기여할지를 정의합니다.
FLAG_WEIGHTS = {
    "F0 변화 폭이 큼": 0.5,
    "F0 상승/하강 기울기 큼": 0.4,
    "Jitter 증가(떨림)": 0.3,
    "Shimmer 증가(음량 미세 변동)": 0.1,
    "HNR 낮음(잡음↑)": 0.1,
    "음량 피크 빈도 증가": 0.25,
    "초당 발성 구간 수 많음": 0.05,
    "발성 구간 평균 길이가 매우 짧음": 0.05,
    "복합 신호: F0, Jitter, 음량 피크 동시 증가": 0.8,
}

# --- 2. 핵심 지표별 임계값 ---
# 각 음성 특징이 '의미 있는' 수준인지를 판단하는 기준값입니다.
THRESH = {
    "F0_mean_high": 30.0,               # F0(음높이) 평균
    "F0_risingSlope_high": 5.0,         # F0 상승 기울기
    "F0_fallingSlope_high": 5.0,        # F0 하강 기울기 (절대값으로 비교)
    "F0_stddevNorm_high": 0.2,          # F0 변화율
    "loudnessPeaksPerSec_high": 2.0,    # 초당 음량 피크 수
    "jitter_high": 0.15,                # 미세한 음높이 떨림
    "shimmer_dB_high": 4.0,             # 미세한 음량 떨림
    "HNR_low": 3.0,                     # 소음 대비 신호 비율 (낮을수록 잡음 많음)
    "voicedSegsPerSec_high": 3.0,       # 초당 발성 구간 수
    "short_voiced_len_sec": 0.25,       # 짧은 발성 구간 길이
}

# --- 3. 분석 헬퍼 함수 ---

def find_col_like(df: pd.DataFrame, *keywords) -> Optional[str]:
    """DataFrame에서 특정 키워드가 포함된 첫 번째 컬럼 이름을 찾습니다."""
    kws = [k.lower() for k in keywords]
    for c in df.columns:
        cl = str(c).lower()
        if all(k in cl for k in kws):
            return c
    return None

def compute_loudness_peaks_per_sec(lld: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """LLD(저수준 기술자) 데이터에서 초당 음량 피크 수를 계산합니다."""
    if lld is None or lld.empty:
        return None, None

    # pyopensmile은 TimedeltaIndex를 사용하므로 시간 정보를 인덱스에서 가져옵니다.
    if isinstance(lld.index, pd.TimedeltaIndex):
        t = lld.index.total_seconds().to_numpy()
    else:
        return None, None

    lcol = find_col_like(lld, "loudness")
    if lcol is None:
        return None, float(t[-1] - t[0]) if len(t) > 1 else 0.0

    y = pd.to_numeric(lld[lcol], errors="coerce").fillna(0.0).values.astype(float)
    if len(y) < 3:
        return 0.0, float(t[-1] - t[0]) if len(t) >= 2 else 0.0

    # 평균+표준편차를 넘는 지점을 피크로 간주
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    thr = mean + std * 1.0

    peaks = 0
    for i in range(1, len(y) - 1):
        if (y[i-1] < y[i] >= y[i+1]) and (y[i] > thr):
            peaks += 1

    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    pps = (peaks / duration) if duration and duration > 0 else 0.0
    return pps, duration

def compute_voicing_segments(lld: pd.DataFrame, prob_thr: float = 0.6) -> Dict[str, Optional[float]]:
    """LLD 데이터에서 발성/무성 구간 관련 통계를 계산합니다."""
    if lld is None or lld.empty:
        return {"DurationSec": None}

    if isinstance(lld.index, pd.TimedeltaIndex):
        t = lld.index.total_seconds().to_numpy()
    else:
        return {"DurationSec": float(lld.shape[0])}

    # 발성 확률 컬럼을 찾습니다.
    vcol = find_col_like(lld, "voicingfinal") or find_col_like(lld, "voicing") or find_col_like(lld, "f0")
    if vcol is None:
        return {"DurationSec": float(t[-1] - t[0]) if len(t) > 1 else 0.0}

    v = pd.to_numeric(lld[vcol], errors="coerce").fillna(0.0).values.astype(float)
    v = (v >= prob_thr).astype(float) # 확률을 이진값(0 또는 1)으로 변환

    if len(v) < 2:
        return {"DurationSec": float(t[-1] - t[0]) if len(t) > 1 else 0.0}

    # 발성/무성 구간을 나누고 길이를 계산합니다.
    segments, voiced_lens, unvoiced_lens = [], [], []
    cur_state, start_idx, voiced_count = v[0], 0, 0
    for i in range(1, len(v)):
        if v[i] != cur_state:
            segments.append((t[start_idx], t[i-1], bool(cur_state)))
            start_idx = i
            cur_state = v[i]
    segments.append((t[start_idx], t[-1], bool(cur_state)))

    for s, e, is_voiced in segments:
        length = max(0.0, float(e - s))
        if is_voiced:
            voiced_lens.append(length)
            voiced_count += 1
        else:
            unvoiced_lens.append(length)

    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    def stats(arr):
        return (float(np.nanmean(arr)), float(np.nanstd(arr))) if arr else (None, None)

    mean_v, std_v = stats(voiced_lens)
    mean_u, std_u = stats(unvoiced_lens)
    vps = (voiced_count / duration) if duration and duration > 0 else 0.0

    return {
        "VoicedSegmentsPerSec": vps, "MeanVoicedSegmentLengthSec": mean_v,
        "StddevVoicedSegmentLengthSec": std_v, "MeanUnvoicedSegmentLengthSec": mean_u,
        "StddevUnvoicedSegmentLengthSec": std_u, "DurationSec": duration,
    }

def safe_get(s: pd.Series, key: str) -> Optional[float]:
    """Series에서 키에 해당하는 값을 float으로 안전하게 가져옵니다."""
    try:
        return float(s[key])
    except (KeyError, ValueError, TypeError):
        return None

def build_summary(func: pd.Series, lld: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    추출된 Functionals(func)와 LLD(lld)를 종합하여 분석 요약본과 플래그를 생성합니다.
    """
    out: Dict[str, Any] = {}

    # 1. Functionals에서 직접 주요 특징 추출
    out["F0_mean_semitone"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_amean")
    out["F0_stddevNorm"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    out["F0_meanRisingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")
    out["F0_meanFallingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope")
    out["jitterLocal_amean"] = safe_get(func, "jitterLocal_sma3nz_amean")
    out["shimmerLocaldB_amean"] = safe_get(func, "shimmerLocaldB_sma3nz_amean")
    out["HNRdBACF_amean"] = safe_get(func, "HNRdBACF_sma3nz_amean")
    out["loudness_amean"] = safe_get(func, "loudness_sma3_amean")

    # 2. LLD에서 파생 특징 계산
    peaks_per_sec, dur = compute_loudness_peaks_per_sec(lld) if lld is not None else (None, None)
    seg_dict = compute_voicing_segments(lld) if lld is not None else {"DurationSec": dur}
    
    out["DurationSec"] = seg_dict.get("DurationSec", dur)
    out["loudnessPeaksPerSec"] = peaks_per_sec
    out.update(seg_dict)

    # 3. 규칙 기반 플래그 생성
    flags = []
    def add_flag(cond: bool, msg: str):
        if cond: flags.append(msg)

    add_flag((out.get("F0_stddevNorm") or 0) >= THRESH["F0_stddevNorm_high"], "F0 변화 폭이 큼")
    add_flag((out.get("F0_meanRisingSlope") or 0) >= THRESH["F0_risingSlope_high"], "F0 상승 기울기 큼")
    add_flag(abs(out.get("F0_meanFallingSlope") or 0) >= THRESH["F0_fallingSlope_high"], "F0 하강 기울기 큼")
    add_flag((out.get("jitterLocal_amean") or 0) >= THRESH["jitter_high"], "Jitter 증가(떨림)")
    add_flag((out.get("shimmerLocaldB_amean") or 0) >= THRESH["shimmer_dB_high"], "Shimmer 증가(음량 미세 변동)")
    add_flag(out.get("HNRdBACF_amean") is not None and out["HNRdBACF_amean"] <= THRESH["HNR_low"], "HNR 낮음(잡음↑)")
    add_flag((out.get("loudnessPeaksPerSec") or 0) >= THRESH["loudnessPeaksPerSec_high"], "음량 피크 빈도 증가")
    add_flag((out.get("VoicedSegmentsPerSec") or 0) >= THRESH["voicedSegsPerSec_high"], "초당 발성 구간 수 많음")
    mvl = out.get("MeanVoicedSegmentLengthSec")
    add_flag(mvl is not None and mvl <= THRESH["short_voiced_len_sec"], "발성 구간 평균 길이가 매우 짧음")
    out["flags"] = flags

    # 4. 요약 코멘트 생성
    comments = []
    if any("F0" in f for f in flags): comments.append("피치 변동성 큼")
    if any(("Jitter" in f) or ("Shimmer" in f) or ("HNR" in f) for f in flags): comments.append("발성 불안정")
    if any(("음량" in f) or ("발성 구간" in f) for f in flags): comments.append("짧고 강한 발성 패턴")
    out["summary_ko"] = ", ".join(comments) if comments else "안정적"

    return out

def build_sudden_rage_probability(summary: dict) -> dict:
    """
    분석 요약(summary)을 바탕으로 최종 '급발진 확률'을 계산합니다.
    None/NaN 방지를 위해 모든 값은 num()으로 정규화 후 사용.
    """
    score = 0.0
    flags = summary.get("flags", [])

    # F0 변화 폭
    f0_std = num(summary.get("F0_stddevNorm"), 0.0)
    if f0_std > THRESH["F0_stddevNorm_high"]:
        ratio = f0_std / THRESH["F0_stddevNorm_high"]
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("F0 변화 폭이 큼", 0.5)

    # F0 상승 기울기
    rising_slope = num(summary.get("F0_meanRisingSlope"), 0.0)
    if rising_slope > THRESH["F0_risingSlope_high"]:
        score += min(rising_slope / THRESH["F0_risingSlope_high"], 5.0) * \
                FLAG_WEIGHTS.get("F0 상승/하강 기울기 큼", 0.4)

    # 음량 피크 빈도
    lpps = num(summary.get("loudnessPeaksPerSec"), 0.0)
    if lpps > THRESH["loudnessPeaksPerSec_high"]:
        ratio = lpps / THRESH["loudnessPeaksPerSec_high"]
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("음량 피크 빈도 증가", 0.25)

    # 복합 조건 (세 값 모두 유의할 때)
    if (f0_std > THRESH["F0_stddevNorm_high"] and
        num(summary.get("jitterLocal_amean"), 0.0) > THRESH["jitter_high"] and
        lpps > THRESH["loudnessPeaksPerSec_high"]):
        score += FLAG_WEIGHTS.get("복합 신호: F0, Jitter, 음량 피크 동시 증가", 0.8)

    probability = min(score, 1.0) * 100

    return {
        "sudden_rage_probability_percent": round(probability, 2),
        "active_flags": list(set(flags)),  # 중복 제거
    }


# ======================================================================
# realtime_analyzer.py에서 가져온 실시간 처리 로직 (수정됨)
# ======================================================================

# --- 오디오 장치 설정 ---
NAME_HINTS = ("RODE", "RØDE", "VIDEOMIC", "USB") # 마이크 이름에 포함될 키워드
SAMPLE_RATE = 48000  # 샘플링 속도 (Hz)
CHANNELS = 1         # 채널 수 (모노)
BUFFER_SECONDS = 2   # 분석에 사용할 오디오 버퍼 길이 (초)
FRAME_HZ = 2         # 초당 분석 횟수 (CPU 사용량 고려하여 조정)
BLOCKSIZE = 2048     # 한 번에 읽어들일 오디오 프레임 크기
SILENCE_RMS_THRESH = 5e-5 # 무음으로 판단할 RMS 임계값

def _device_name(idx: Optional[int]) -> str:
    """장치 인덱스로부터 장치 이름을 가져옵니다."""
    try:
        if idx is None: return "(default)"
        return sd.query_devices(idx)["name"]
    except Exception:
        return "(unknown)"

def pick_input_device(name_hints=NAME_HINTS) -> Optional[int]:
    """지정된 키워드와 일치하는 첫 번째 입력 장치를 찾습니다."""
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                if any(h in d.get("name", "").upper() for h in name_hints):
                    return i
        # 힌트와 맞는 장치가 없으면 첫 번째 입력 장치를 사용
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                return i
    except Exception as e:
        print(f"[에러] 오디오 장치를 찾을 수 없습니다: {e}", file=sys.stderr)
    return None

# --- 실시간 분석 클래스 ---
class RealTimeRageDetector:
    """
    백그라운드 스레드에서 실시간으로 마이크 입력을 받아 openSMILE로 분석하고,
    급발진 확률을 계산하여 제공하는 클래스.
    """
    def __init__(self, device_index: Optional[int], **kwargs) -> None:
        self.device = device_index
        self.samplerate = int(kwargs.get("samplerate", SAMPLE_RATE))
        buffer_seconds = int(kwargs.get("buffer_seconds", BUFFER_SECONDS))
        self.buffer_samples = self.samplerate * buffer_seconds
        self.frame_hz = float(kwargs.get("frame_hz", FRAME_HZ))
        self.silence_rms_thresh = float(kwargs.get("silence_rms_thresh", SILENCE_RMS_THRESH))
        
        self._buf: deque[float] = deque(maxlen=self.buffer_samples)
        self._buf_lock = threading.Lock()

        # --- openSMILE 인스턴스 생성 ---
        # 1) Functionals OK
        self.smile_func = pyopensmile.Smile(
            feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
            feature_level=pyopensmile.FeatureLevel.Functionals,
        )

        # 2) LLD -> LowLevelDescriptors 로 변경
        self.smile_lld = pyopensmile.Smile(
            feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
            feature_level=pyopensmile.FeatureLevel.LowLevelDescriptors,  # ✅ 요걸로
        )


        self._latest: Optional[Dict[str, Any]] = None
        self._latest_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """분석 스레드를 시작합니다."""
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """분석 스레드를 정지합니다."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)

    def read(self) -> Optional[Dict[str, Any]]:
        """가장 최근의 분석 결과를 반환합니다."""
        with self._latest_lock:
            return dict(self._latest) if self._latest else None

    def _audio_callback(self, indata, frames, time_info, status):
        """오디오 스트림에서 호출되는 콜백. 버퍼에 데이터를 추가합니다."""
        if status: print(status, file=sys.stderr)
        with self._buf_lock:
            self._buf.extend(indata[:, 0])

    def _loop(self):
        """메인 분석 루프 (백그라운드 스레드에서 실행)."""
        dev_name = _device_name(self.device)
        print(f"[정보] 입력 장치: '{dev_name}' (샘플링 속도: {self.samplerate}Hz)")
        
        stream_kwargs = dict(
            samplerate=self.samplerate, channels=CHANNELS,
            blocksize=BLOCKSIZE, dtype="float32",
            callback=self._audio_callback, latency="low", device=self.device
        )
        
        with sd.InputStream(**stream_kwargs):
            print(f"[정보] {BUFFER_SECONDS}초 분량의 오디오 버퍼를 채우는 중...")
            while self._running and len(self._buf) < self.buffer_samples:
                time.sleep(0.1)
            
            print("[정보] 버퍼 준비 완료. 실시간 분석을 시작합니다.")
            next_time = time.monotonic()
            rms_tail = max(1, int(self.samplerate * 0.5)) # 0.5초 분량 RMS 계산

            while self._running:
                now = time.monotonic()
                if now < next_time:
                    time.sleep(0.005)
                    continue

                # --- 1. 오디오 버퍼 복사 ---
                with self._buf_lock:
                    if len(self._buf) < self.buffer_samples: continue
                    audio = np.array(self._buf, dtype=np.float32)

                # --- 2. 무음 구간 건너뛰기 ---
                rms = np.sqrt(np.mean(audio[-rms_tail:]**2))
                if rms < self.silence_rms_thresh:
                    next_time += 1.0 / self.frame_hz
                    continue

                # --- 3. openSMILE 특징 추출 ---
                try:
                    # Functionals와 LLD를 모두 추출합니다.
                    func_df = self.smile_func.process_signal(audio, self.samplerate)
                    lld_df = self.smile_lld.process_signal(audio, self.samplerate)
                    
                    # --- 4. 요약 및 확률 계산 ---
                    summary = build_summary(func_df.iloc[0], lld_df)
                    rage_result = build_sudden_rage_probability(summary)
                    summary.update(rage_result)

                    # --- 5. 최신 결과 저장 ---
                    with self._latest_lock:
                        self._latest = summary
                except Exception as e:
                    print(f"[에러] 특징 추출 또는 분석 중 오류 발생: {e}", file=sys.stderr)
                
                next_time += 1.0 / self.frame_hz

    def run_in_console(self) -> None:
        """콘솔에 분석 결과를 계속 출력하는 메서드."""
        self.start()
        try:
            while True:
                result = self.read()
                if result:
                    prob = result.get('sudden_rage_probability_percent', 0)
                    summary = result.get('summary_ko', '')
                    flags = result.get('active_flags', [])
                    # 을 사용하여 한 줄에 덮어쓰기
                    print(f"확률: {prob:5.1f}% | 상태: {summary:<15} | 감지된 플래그: {flags}      ", end="")
                else:
                    print(f"분석 준비 중...", end="")
                time.sleep(1.0 / self.frame_hz)
        except KeyboardInterrupt:
            print("[정보] 분석을 중지합니다.")
        finally:
            self.stop()

# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    # 1. 사용 가능한 오디오 입력 장치 선택
    input_device_index = pick_input_device()
    
    if input_device_index is None:
        print("[치명적 에러] 사용 가능한 오디오 입력 장치를 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    # 2. 분석기 인스턴스 생성
    analyzer = RealTimeRageDetector(
        device_index=input_device_index,
        samplerate=SAMPLE_RATE,
        buffer_seconds=BUFFER_SECONDS,
        frame_hz=FRAME_HZ,
        silence_rms_thresh=SILENCE_RMS_THRESH,
    )
    
    # 3. 콘솔에서 실행
    analyzer.run_in_console()
