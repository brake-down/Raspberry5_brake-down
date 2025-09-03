# 민서 

import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import arff  

# ===================== 사용자 경로 설정 =====================
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
SMILE_PATH  = r"C:\Users\codin\OneDrive\바탕 화면\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
CONFIG_PATH = r"C:\Users\codin\OneDrive\바탕 화면\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\gemaps\v01b\GeMAPSv01b.conf"

# 분석할 영상 파일(이 이름을 기준으로 출력 파일 자동 생성)
VIDEO_PATH  = r"C:\Users\codin\OneDrive\바탕 화면\sample11.mp4"

video_basename = os.path.basename(VIDEO_PATH)
stem, _ = os.path.splitext(video_basename)
base_dir = os.path.dirname(VIDEO_PATH)

audio_path  = os.path.join(base_dir,"opensmile-3.0.2-windows-x86_64", "데이터", f"{stem}.wav")
func_csv    = os.path.join(base_dir, "opensmile-3.0.2-windows-x86_64", "데이터",f"{stem}_func.csv")
lld_csv     = os.path.join(base_dir,"opensmile-3.0.2-windows-x86_64", "데이터", f"{stem}_lld.csv")
report_json = os.path.join(base_dir,"opensmile-3.0.2-windows-x86_64", "데이터", f"{stem}_report.json")
report_md   = os.path.join(base_dir, "opensmile-3.0.2-windows-x86_64", "데이터", f"{stem}_report.md")
# ===================== 유틸: openSMILE ARFF -> DataFrame =====================
def arff_to_dataframe(path: str) -> pd.DataFrame:
    """
    openSMILE Functional 출력(ARFF 헤더가 들어간 .csv 포함)을 pandas DataFrame으로 변환.
    - liac-arff(arff.load)로 파싱
    - 'numeric' 속성을 컬럼으로 사용
    - '?', 'unknown', 빈값은 NaN으로 들어오므로 이후 float 변환 시 coerce
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = arff.load(f)

    # attributes: [(name, type), ...]  type은 'NUMERIC' 등
    attr_names = [a[0] for a in data["attributes"]]
    df = pd.DataFrame(data["data"], columns=attr_names)

    # 숫자 컬럼들을 최대한 float로 캐스팅
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ===================== 1) FFMPEG: 영상→오디오 =====================
def extract_wav(ffmpeg: str, video: str, wav: str, sr: int = 16000, mono: bool = True) -> None:
    if not os.path.isfile(video):
        raise FileNotFoundError(f"영상 파일을 찾지 못했습니다: {video}")
    os.makedirs(os.path.dirname(wav), exist_ok=True)

    cmd = [ffmpeg, "-y", "-i", video, "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [wav]

    print("[1] 영상에서 오디오 추출 중...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("ffmpeg 변환 실패")
    print(f"[완료] 오디오 추출됨: {wav}")

# ===================== 2) openSMILE 실행 =====================
def run_opensmile(smile: str, config: str, audio: str, func_csv: str, lld_csv: Optional[str] = None) -> None:
    if not os.path.isfile(audio):
        raise FileNotFoundError(f"오디오 파일을 찾지 못했습니다: {audio}")
    os.makedirs(os.path.dirname(func_csv), exist_ok=True)
    if lld_csv:
        os.makedirs(os.path.dirname(lld_csv), exist_ok=True)

    cmd = [
        smile,
        "-C", config,
        "-I", audio,
        "-O", func_csv,          # Functional (ARFF 헤더 포함)
        "-loglevel", "1",
    ]
    if lld_csv:
        cmd += ["-lldcsvoutput", lld_csv]  # LLD CSV(세미콜론 구분) 요청

    print("[2] openSMILE 실행 중...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("openSMILE 실행 실패")
    print(f"[완료] 기능 CSV(ARFF): {func_csv}")
    if lld_csv and os.path.isfile(lld_csv):
        print(f"[완료] LLD CSV: {lld_csv}")
    elif lld_csv:
        print("[경고] LLD CSV가 생성되지 않았습니다. config/버전 차이일 수 있습니다.")

# ===================== 3) 기능 CSV 파싱 & 선택 =====================
def parse_functionals(func_csv: str) -> pd.Series:
    """
    openSMILE Functional 결과(ARFF 헤더 포함)를 liac-arff로 읽고
    첫 행을 Series로 반환. (GeMAPSv01b는 보통 단일 행)
    """
    df = arff_to_dataframe(func_csv)

    # 딱 우리가 원하는 방식: F0semitoneFrom27.5Hz_sma3nz_amean 같은 컬럼이 곧바로 존재
    if df.shape[0] >= 1:
        return df.iloc[0]
    return pd.Series(dtype=float)

# ===================== 4) LLD 기반 파생 지표 계산 =====================
def find_col_like(df: pd.DataFrame, *keywords) -> Optional[str]:
    kws = [k.lower() for k in keywords]
    for c in df.columns:
        cl = str(c).lower()
        if all(k in cl for k in kws):
            return c
    return None

def compute_loudness_peaks_per_sec(lld: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """loudness LLD에서 로컬 피크 개수를 초당 개수로 환산. 반환: (peaks_per_sec, duration_sec)"""
    if lld is None or lld.empty:
        return None, None

    tcol = find_col_like(lld, "frame", "time")
    if tcol is None:
        return None, None

    lcol = find_col_like(lld, "loudness")
    if lcol is None:
        return None, float(lld.shape[0])

    t = pd.to_numeric(lld[tcol], errors="coerce").fillna(0.0).values.astype(float)
    y = pd.to_numeric(lld[lcol], errors="coerce").fillna(0.0).values.astype(float)
    if len(y) < 3:
        return 0.0, float(t[-1] - t[0]) if len(t) >= 2 else None

    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    thr = mean + std * 1.0

    peaks = 0
    for i in range(1, len(y) - 1):
        if (y[i-1] < y[i] >= y[i+1]) and (y[i] > thr):
            peaks += 1

    duration = float(t[-1] - t[0]) if len(t) >= 2 else None
    pps = (peaks / duration) if duration and duration > 0 else None
    return pps, duration

def compute_voicing_segments(lld: pd.DataFrame, prob_thr: float = 0.6) -> Dict[str, Optional[float]]:
    """voicingFinalUnclipped(또는 유사)로 발성/무성 세그먼트 통계 계산"""
    if lld is None or lld.empty:
        return {
            "VoicedSegmentsPerSec": None,
            "MeanVoicedSegmentLengthSec": None,
            "StddevVoicedSegmentLengthSec": None,
            "MeanUnvoicedSegmentLengthSec": None,
            "StddevUnvoicedSegmentLengthSec": None,
            "DurationSec": None,
        }

    tcol = find_col_like(lld, "frame", "time")
    vcol = find_col_like(lld, "voicingfinal") or find_col_like(lld, "voicing") or find_col_like(lld, "f0")
    if tcol is None or vcol is None:
        return {
            "VoicedSegmentsPerSec": None,
            "MeanVoicedSegmentLengthSec": None,
            "StddevVoicedSegmentLengthSec": None,
            "MeanUnvoicedSegmentLengthSec": None,
            "StddevUnvoicedSegmentLengthSec": None,
            "DurationSec": float(lld.shape[0]),
        }

    t = pd.to_numeric(lld[tcol], errors="coerce").fillna(0.0).values.astype(float)
    v = pd.to_numeric(lld[vcol], errors="coerce").fillna(0.0).values.astype(float)

    if "voicing" not in str(vcol).lower():
        v = (v > 0).astype(float)
    else:
        v = (v >= prob_thr).astype(float)

    segments = []
    cur_state = v[0]
    start_idx = 0
    for i in range(1, len(v)):
        if v[i] != cur_state:
            segments.append((t[start_idx], t[i-1], bool(cur_state)))
            start_idx = i
            cur_state = v[i]
    segments.append((t[start_idx], t[-1], bool(cur_state)))

    voiced_lens, unvoiced_lens = [], []
    voiced_count = 0
    for s, e, is_voiced in segments:
        length = max(0.0, float(e - s))
        if is_voiced:
            voiced_lens.append(length)
            voiced_count += 1
        else:
            unvoiced_lens.append(length)

    duration = float(t[-1] - t[0]) if len(t) >= 2 else None

    def stats(arr):
        if len(arr) == 0:
            return (None, None)
        return (float(np.nanmean(arr)), float(np.nanstd(arr)))

    mean_v, std_v = stats(voiced_lens)
    mean_u, std_u = stats(unvoiced_lens)

    vps = (voiced_count / duration) if duration and duration > 0 else None

    return {
        "VoicedSegmentsPerSec": vps,
        "MeanVoicedSegmentLengthSec": mean_v,
        "StddevVoicedSegmentLengthSec": std_v,
        "MeanUnvoicedSegmentLengthSec": mean_u,
        "StddevUnvoicedSegmentLengthSec": std_u,
        "DurationSec": duration,
    }

# ===================== 5) 요약/플래그/리포트 =====================
def safe_get(s: pd.Series, key: str) -> Optional[float]:
    try:
        return float(s[key])
    except Exception:
        return None

def build_summary(func: pd.Series, lld: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # LLD에서 F0 평균 (semitone) 추정
    f0_mean = None
    if lld is not None and not lld.empty:
        f0_col = find_col_like(lld, "f0semitone")
        if f0_col:
            f0_values = pd.to_numeric(lld[f0_col], errors="coerce").fillna(0.0).values
            f0_nonzero = f0_values[f0_values > 0]
            if len(f0_nonzero) > 0:
                f0_mean = float(np.mean(f0_nonzero))
    out["F0_mean_lld"] = f0_mean

    # Functional(ARFF)에서 직접 읽기 — 사용자가 원하는 방식 유지
    out["F0_mean_semitone"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_amean")
    out["F0_stddevNorm"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    out["F0_meanRisingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")
    out["F0_meanFallingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope")

    out["jitterLocal_amean"] = safe_get(func, "jitterLocal_sma3nz_amean")
    out["shimmerLocaldB_amean"] = safe_get(func, "shimmerLocaldB_sma3nz_amean")
    out["HNRdBACF_amean"] = safe_get(func, "HNRdBACF_sma3nz_amean")

    out["loudness_amean"] = safe_get(func, "loudness_sma3_amean")
    out["loudness_stddevNorm"] = safe_get(func, "loudness_sma3_stddevNorm")
    out["loudness_percentile50"] = safe_get(func, "loudness_sma3_percentile50.0")

    out["alphaRatioV"] = safe_get(func, "alphaRatioV_sma3nz_amean")
    out["alphaRatioUV"] = safe_get(func, "alphaRatioUV_sma3nz_amean")
    out["hammarbergIndexV"] = safe_get(func, "hammarbergIndexV_sma3nz_amean")
    out["hammarbergIndexUV"] = safe_get(func, "hammarbergIndexUV_sma3nz_amean")
    out["slopeV0_500"] = safe_get(func, "slopeV0-500_sma3nz_amean")
    out["slopeV500_1500"] = safe_get(func, "slopeV500-1500_sma3nz_amean")

    # 2) LLD 파생 지표
    peaks_per_sec, dur = compute_loudness_peaks_per_sec(lld) if lld is not None else (None, None)
    seg = compute_voicing_segments(lld) if lld is not None else {
        "VoicedSegmentsPerSec": None,
        "MeanVoicedSegmentLengthSec": None,
        "StddevVoicedSegmentLengthSec": None,
        "MeanUnvoicedSegmentLengthSec": None,
        "StddevUnvoicedSegmentLengthSec": None,
        "DurationSec": dur,
    }

    out["DurationSec"] = seg.get("DurationSec", dur)
    out["loudnessPeaksPerSec"] = peaks_per_sec
    out.update({
        "VoicedSegmentsPerSec": seg["VoicedSegmentsPerSec"],
        "MeanVoicedSegmentLengthSec": seg["MeanVoicedSegmentLengthSec"],
        "StddevVoicedSegmentLengthSec": seg["StddevVoicedSegmentLengthSec"],
        "MeanUnvoicedSegmentLengthSec": seg["MeanUnvoicedSegmentLengthSec"],
        "StddevUnvoicedSegmentLengthSec": seg["StddevUnvoicedSegmentLengthSec"],
    })

    # 3) 규칙 기반 플래그
    flags = []

    is_f0_big = (out.get("F0_stddevNorm") or 0) >= THRESH["F0_stddevNorm_high"]
    is_jitter_big = (out.get("jitterLocal_amean") or 0) >= THRESH["jitter_high"]
    is_loud_peak = (out.get("loudnessPeaksPerSec") or 0) >= THRESH["loudnessPeaksPerSec_high"]
    
    if is_f0_big and is_jitter_big and is_loud_peak:
        flags.append("복합 신호: F0, Jitter, 음량 피크 동시 증가")

    def add_flag(cond: bool, msg: str):
        if cond:
            flags.append(msg)

    add_flag(is_f0_big, "F0 변화 폭이 큼")

    rs = out.get("F0_meanRisingSlope")
    fs = out.get("F0_meanFallingSlope")
    add_flag(rs is not None and abs(rs) >= abs(THRESH["F0_risingSlope_high"]), "F0 상승/하강 기울기 큼")
    add_flag(fs is not None and abs(fs) >= abs(THRESH["F0_fallingSlope_high"]), "F0 상승/하강 기울기 큼")

    add_flag((out.get("jitterLocal_amean") or 0) >= THRESH["jitter_high"], "Jitter 증가(떨림)")
    add_flag((out.get("shimmerLocaldB_amean") or 0) >= THRESH["shimmer_dB_high"], "Shimmer 증가(음량 미세 변동)")
    hnr = out.get("HNRdBACF_amean")
    add_flag(hnr is not None and hnr <= THRESH["HNR_low"], "HNR 낮음(잡음↑)")

    add_flag((out.get("loudnessPeaksPerSec") or 0) >= THRESH["loudnessPeaksPerSec_high"], "음량 피크 빈도 증가")
    add_flag((out.get("VoicedSegmentsPerSec") or 0) >= THRESH["voicedSegsPerSec_high"], "초당 발성 구간 수 많음")

    mvl = out.get("MeanVoicedSegmentLengthSec")
    add_flag(mvl is not None and mvl <= THRESH["short_voiced_len_sec"], "발성 구간 평균 길이가 매우 짧음")

    out["flags"] = flags

    # 4) 요약 코멘트
    comments = []
    if any("F0" in f for f in flags):
        comments.append("피치 관련 변동이 커 당황/비명 가능성이 있습니다.")
    if any(("Jitter" in f) or ("Shimmer" in f) or ("HNR" in f) for f in flags):
        comments.append("발성 안정성이 낮아 긴장/불안정 음성일 수 있습니다.")
    if any(("음량 피크" in f) or ("발성 구간" in f) for f in flags):
        comments.append("큰 소리/짧은 반복 발성 패턴이 관찰됩니다.")
    out["summary_ko"] = " ".join(comments) if comments else "특이 신호가 두드러지지 않습니다 (임계값은 데이터에 맞게 보정 필요)."

    return out

def build_report(summary: dict) -> str:
    # 사용자가 원한 Markdown 형식 그대로
    lines = [
        "# 음성 특징 기반 급발진/긴장 신호 분석 요약",
        "",
        f"- 총 길이(DurationSec): {summary.get('DurationSec')} s",
        "",
        "## 핵심 지표",
        f"- F0 평균(semitone): {summary.get('F0_mean_semitone')}",
        f"- F0 변화 폭(stddevNorm): {summary.get('F0_stddevNorm')}",
        f"- F0 상승/하강 기울기: {summary.get('F0_meanRisingSlope')} / {summary.get('F0_meanFallingSlope')}",
        f"- Jitter/Shimmer/HNR: {summary.get('jitterLocal_amean')} / {summary.get('shimmerLocaldB_amean')} / {summary.get('HNRdBACF_amean')} dB",
        f"- Loudness 평균/표준편차정규화/중앙값: {summary.get('loudness_amean')} / {summary.get('loudness_stddevNorm')} / {summary.get('loudness_percentile50')}",
        f"- Loudness 피크/초: {summary.get('loudnessPeaksPerSec')}",
        f"- 초당 발성 구간 수: {summary.get('VoicedSegmentsPerSec')}",
        f"- 발성 구간 길이(평균±표준편차): {summary.get('MeanVoicedSegmentLengthSec')} ± {summary.get('StddevVoicedSegmentLengthSec')} s",
        f"- 무성 구간 길이(평균±표준편차): {summary.get('MeanUnvoicedSegmentLengthSec')} ± {summary.get('StddevUnvoicedSegmentLengthSec')} s",
        "",
        "## 스펙트럼/음색",
        f"- alphaRatio(V/UV): {summary.get('alphaRatioV')} / {summary.get('alphaRatioUV')}",
        f"- hammarbergIndex(V/UV): {summary.get('hammarbergIndexV')} / {summary.get('hammarbergIndexUV')}",
        f"- slopeV(0-500 / 500-1500): {summary.get('slopeV0_500')} / {summary.get('slopeV500_1500')}",
        "",
        "## 플래그",
        "- " + ("\n- ".join(summary.get("flags", []) or ["(없음)"])),
        "",
        "## 해석(요약)",
        f"- {summary.get('summary_ko')}",
        "",
    ]
    return "\n".join(lines)

# ===================== 6) 급발진 확률 계산 =====================
# ===================== 1) 플래그 가중치 정의 (사용은 참고용) =====================
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

# ===================== 2) 핵심 지표 임계값 =====================
THRESH = {
    "F0_mean_high": 30.0,
    "F0_risingSlope_high": 5.0,
    "F0_fallingSlope_high": -5.0,
    "F0_stddevNorm_high": 0.2,
    "F0_slope_high": 250.0,
    "vowel_segment_mean_low": 0.1,
    "loudnessPeaksPerSec_high": 2.0,
    "jitter_high": 0.15,
    "voicedSegsPerSec_high": 3.0,
    "shimmer_dB_high": 4.0,
    "HNR_low": 3.0,
    "short_voiced_len_sec": 0.25
}

# ===================== 3) 확률 계산 함수 수정 =====================
def build_sudden_rage_probability(summary: dict) -> dict:
    """
    summary: build_summary 결과
    핵심 지표 기반으로 급발진 확률 계산
    """
    score = 0.0
    flags = summary.get("flags", [])

    # F0 평균
    if summary.get("F0_mean", 0) > THRESH["F0_mean_high"]:
        score += FLAG_WEIGHTS.get("F0 변화 폭이 큼", 0.5)

    # F0 변화 폭
    if summary.get("F0_stddevNorm", 0) > THRESH["F0_stddevNorm_high"]:
        ratio = summary["F0_stddevNorm"] / THRESH["F0_stddevNorm_high"]
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("F0 변화 폭이 큼", 0.5)

    # F0 상승/하강 기울기
    if (summary.get("F0_slope_up", 0) > THRESH["F0_slope_high"] or 
        summary.get("F0_slope_down", 0) > THRESH["F0_slope_high"]):
        ratio = max(summary.get("F0_slope_up",0), summary.get("F0_slope_down",0)) / THRESH["F0_slope_high"]
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("F0 상승/하강 기울기 큼", 0.4)

    # 발성 구간 평균 길이
    if summary.get("vowel_segment_mean", 1.0) < THRESH["vowel_segment_mean_low"]:
        ratio = THRESH["vowel_segment_mean_low"] / max(summary.get("vowel_segment_mean", 0.01), 0.01)
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("발성 구간 평균 길이가 매우 짧음", 0.3)

    # 음량 피크 빈도
    if summary.get("loudnessPeaksPerSec", 0) > THRESH["loudnessPeaksPerSec_high"]:
        ratio = summary["loudnessPeaksPerSec"] / THRESH["loudnessPeaksPerSec_high"]
        score += min(ratio, 5.0) * FLAG_WEIGHTS.get("음량 피크 빈도 증가", 0.25)

    # 복합 조건: F0, Jitter, Loudness 동시에 증가
    if (summary.get("F0_stddevNorm", 0) > THRESH["F0_stddevNorm_high"] and
        summary.get("jitterLocal_amean", 0) > THRESH["jitter_high"] and
        summary.get("loudnessPeaksPerSec", 0) > THRESH["loudnessPeaksPerSec_high"]):
        score += FLAG_WEIGHTS.get("복합 신호: F0, Jitter, 음량 피크 동시 증가", 0.8)

    # 점수 0~1 제한 후 %
    probability = min(score, 1.0) * 100

    return {
        "sudden_rage_probability_percent": round(probability, 2),
        "active_flags": flags
    }


# ===================== 7) 메인 =====================
def main():
    # 1) 영상 → 오디오
    if not os.path.isfile(audio_path):
        extract_wav(FFMPEG_PATH, VIDEO_PATH, audio_path, sr=16000, mono=True)
    else:
        print(f"[건너뜀] 오디오가 이미 존재: {audio_path}")

    # 2) openSMILE 실행
    run_opensmile(SMILE_PATH, CONFIG_PATH, audio_path, func_csv, lld_csv)

    # 3) 기능(ARFF) 파싱
    func_row = parse_functionals(func_csv)

    # 4) LLD CSV 로드
    lld_df = None
    if os.path.isfile(lld_csv):
        try:
            lld_df = pd.read_csv(lld_csv, sep=";", engine="python")
        except Exception:
            lld_df = pd.read_csv(lld_csv, engine="python")
    if lld_df is not None:
        lld_df = lld_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # 5) 요약 생성
    summary = build_summary(func_row, lld_df)

    # 6) 확률 계산 (새 방식)
    rage_result = build_sudden_rage_probability(summary)
    summary.update(rage_result)

    # 7) 저장
    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_text = build_report(summary)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n=== 급발진 확률 분석 ===")
    print(f"확률: {summary['sudden_rage_probability_percent']}%")
    print("활성 플래그:", summary["active_flags"])
    print(f"\n[완료] JSON 리포트: {report_json}")
    print(f"[완료] Markdown 리포트: {report_md}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[에러]", e)
        sys.exit(1)