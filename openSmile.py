import os
import sys
import json
import math
import subprocess
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

# ===================== 사용자 경로 설정 =====================
FFMPEG_PATH = r"C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe"
SMILE_PATH  = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\opensmile-3.0.2-windows-x86_64\\opensmile-3.0.2-windows-x86_64\\bin\\SMILExtract.exe"
CONFIG_PATH = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\opensmile-3.0.2-windows-x86_64\\opensmile-3.0.2-windows-x86_64\\config\\gemaps\\v01b\\GeMAPSv01b.conf"

VIDEO_PATH  = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\sample1.mp4"  # 입력 영상
AUDIO_PATH  = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\sample1.wav"  # 추출 WAV
FUNC_CSV    = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\output.csv"  # 기능(Functional) 출력
LLD_CSV     = r"C:\Users\codin\OneDrive\바탕 화면\output_lld.csv"  # LLD(프레임별) 출력(가능 시)
REPORT_JSON = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\analysis_report.json"
REPORT_MD   = r"C:\\Users\\codin\\OneDrive\\바탕 화면\\analysis_report.md"

# ===================== 분석 임계값(조정 권장) =====================
THRESH = {
    # Pitch/F0 (단위: semitone)
    "F0_stddevNorm_high": 10.0,  # 피치 변화 폭이 큰 편
    "F0_risingSlope_high": 0.5,
    "F0_fallingSlope_high": -0.5,  # 절댓값으로 판단할 것

    # Voice stability
    "jitter_high": 0.015,         # 0.01~0.02 부근에서 민감
    "shimmer_dB_high": 0.4,       # 0.3~0.5 dB 이상 유의
    "HNR_low": 10.0,              # dB, 낮을수록 잡음↑

    # Loudness/Segments
    "loudnessPeaksPerSec_high": 1.5,
    "voicedSegsPerSec_high": 3.0,

    # 세그먼트 길이(초)
    "short_voiced_len_sec": 0.25,  # 매우 짧은 비명/단속 발성
}

# ===================== 유틸: openSMILE ARFF 포맷 CSV 로더 =====================
def parse_arff(path: str) -> pd.DataFrame:
    """
    openSMILE ARFF 파일을 pandas DataFrame으로 변환
    - numeric 속성만 컬럼으로 사용
    - 결측치, 'unknown', '?' 등은 0.0으로 처리
    - 세미콜론, 탭, 콤마 구분자 모두 지원
    - 속성 개수와 값 개수 불일치 시 자동 맞춤 (잘라내거나 0.0 패딩)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # @data 위치 찾기
    data_index = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("@data"):
            data_index = i + 1
            break

    if data_index is None:
        raise RuntimeError(f"ARFF 포맷 오류: {path} (데이터 섹션을 찾을 수 없음)")

    # numeric attribute 이름들 추출
    attributes = [
        line.split()[1]
        for line in lines
        if line.strip().lower().startswith("@attribute") and "numeric" in line.lower()
    ]

    # 데이터 파싱
    data_lines = lines[data_index:]
    values_list = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        # 여러 구분자 처리
        if "\t" in line:
            raw_values = line.split("\t")
        elif ";" in line:
            raw_values = line.split(";")
        else:
            raw_values = line.split(",")

        values = []
        for x in raw_values:
            x = x.strip().strip("'\"")
            if x == "" or x.lower() == "unknown" or x == "?":
                values.append(0.0)
            else:
                try:
                    values.append(float(x))
                except ValueError:
                    values.append(0.0)

        # 속성 개수와 다르면 자동 맞춤
        if len(values) < len(attributes):
            values.extend([0.0] * (len(attributes) - len(values)))
        elif len(values) > len(attributes):
            values = values[:len(attributes)]

        values_list.append(values)

    return pd.DataFrame(values_list, columns=attributes)

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
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
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

    # 기본 기능(Functional) CSV 출력
    cmd = [
        smile,
        "-C", config,
        "-I", audio,
        "-O", func_csv,
        "-loglevel", "1",
    ]

    # 가능 시 LLD CSV 동시 출력 (openSMILE 대부분 버전에서 지원)
    if lld_csv:
        cmd += ["-lldcsvoutput", lld_csv]

    print("[2] openSMILE 실행 중...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("openSMILE 실행 실패")

    print(f"[완료] 기능 CSV: {func_csv}")
    if lld_csv and os.path.isfile(lld_csv):
        print(f"[완료] LLD CSV: {lld_csv}")
    elif lld_csv:
        print("[경고] LLD CSV가 생성되지 않았습니다. config/버전 차이일 수 있습니다.")

# ===================== 3) 기능 CSV 파싱 & 선택 =====================
FEATURE_KEYS = [
    # Pitch / F0
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",

    # Jitter / Shimmer / HNR
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",

    # Loudness (일부 이름은 버전에 따라 다를 수 있어 후처리로 보강)
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile50.0",

    # Spectral / Timbre (존재하는 경우만 취득)
    "alphaRatioV",
    "alphaRatioUV",
    "hammarbergIndexV",
    "hammarbergIndexUV",
    "slopeV0-500",
    "slopeV500-1500",

    # Segment (있을 수도, 없을 수도)
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLengthSec",
    "StddevUnvoicedSegmentLengthSec",
]


def parse_functionals(func_csv: str) -> pd.Series:
    df = parse_arff(func_csv)

    # 케이스 A) 너비형(가로로 feature명 컬럼) → 첫 행이 값
    if df.shape[0] == 1 and df.shape[1] > 2 and df.columns.nlevels == 1:
        row = df.iloc[0]
        return row

    # 케이스 B) 세로형(name, value) → 피벗
    lowered = [c.lower() for c in df.columns]
    if set(["name", "value"]).issubset(set(lowered)) and df.shape[1] >= 2:
        # 실제 컬럼명 찾기
        name_col = df.columns[lowered.index("name")]
        val_col = df.columns[lowered.index("value")]
        s = df.set_index(name_col)[val_col]
        s.name = 0
        return s

    # 기타 포맷은 첫 행 기준으로 처리(최선)
    return df.iloc[0]

# ===================== 4) LLD 기반 파생 지표 계산 =====================
def find_col_like(df: pd.DataFrame, *keywords) -> Optional[str]:
    kws = [k.lower() for k in keywords]
    for c in df.columns:
        cl = str(c).lower()
        if all(k in cl for k in kws):
            return c
    return None


def compute_loudness_peaks_per_sec(lld: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """loudness LLD에서 로컬 피크 개수를 초당 개수로 환산.
    반환: (peaks_per_sec, duration_sec)
    """
    if lld is None or lld.empty:
        return None, None

    # frameTime 찾기
    tcol = find_col_like(lld, "frame", "time")
    if tcol is None:
        # 시간 정보가 없으면 추정 불가
        return None, None

    # loudness 컬럼 찾기 (예: loudness_sma3)
    lcol = find_col_like(lld, "loudness")
    if lcol is None:
        return None, float(lld.shape[0])  # 대략 프레임 수 반환

    t = lld[tcol].values.astype(float)
    y = lld[lcol].values.astype(float)
    if len(y) < 3:
        return 0.0, float(t[-1] - t[0]) if len(t) >= 2 else None

    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    thr = mean + std * 1.0  # 보수적 임계치(조정 가능)

    # 로컬 피크: y[i-1] < y[i] >= y[i+1] & y[i] > thr
    peaks = 0
    for i in range(1, len(y) - 1):
        if (y[i-1] < y[i] >= y[i+1]) and (y[i] > thr):
            peaks += 1

    duration = float(t[-1] - t[0]) if len(t) >= 2 else None
    pps = (peaks / duration) if duration and duration > 0 else None
    return pps, duration


def compute_voicing_segments(lld: pd.DataFrame, prob_thr: float = 0.6) -> Dict[str, Optional[float]]:
    """voicingFinalUnclipped (또는 유사)로 발성/무성 세그먼트 통계 계산"""
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

    t = lld[tcol].values.astype(float)
    v = lld[vcol].values.astype(float)

    # voicing 확률이 아닌 경우(F0 semitone 등): F0 > 0 기준으로 가설 분류
    if "voicing" not in str(vcol).lower():
        v = (v > 0).astype(float)
    else:
        v = (v >= prob_thr).astype(float)

    # 구간 변화를 이용해 세그먼트 산출
    segments = []  # (start_t, end_t, voiced: bool)
    cur_state = v[0]
    start_idx = 0
    for i in range(1, len(v)):
        if v[i] != cur_state:
            segments.append((t[start_idx], t[i-1], bool(cur_state)))
            start_idx = i
            cur_state = v[i]
    # 마지막 구간
    segments.append((t[start_idx], t[-1], bool(cur_state)))

    voiced_lens = []
    unvoiced_lens = []
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
def safe_get(s: pd.Series, key: str, alt_search: Optional[Tuple[str, ...]] = None) -> Optional[float]:
    # 정확 키 우선
    if key in s.index:
        try:
            return float(s[key])
        except Exception:
            return None
    # 대체 패턴 검색
    if alt_search:
        for pat in alt_search:
            for col in s.index:
                if pat.lower() in str(col).lower():
                    try:
                        return float(s[col])
                    except Exception:
                        pass
    return None


def build_summary(func: pd.Series, lld: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1) 기능값 추출 (존재 시)
    out["F0_mean_semitone"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_amean")
    out["F0_stddevNorm"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
    out["F0_meanRisingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope")
    out["F0_meanFallingSlope"] = safe_get(func, "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope")

    out["jitterLocal_amean"] = safe_get(func, "jitterLocal_sma3nz_amean")
    out["shimmerLocaldB_amean"] = safe_get(func, "shimmerLocaldB_sma3nz_amean")
    out["HNRdBACF_amean"] = safe_get(func, "HNRdBACF_sma3nz_amean")

    out["loudness_amean"] = safe_get(func, "loudness_sma3_amean")
    out["loudness_stddevNorm"] = safe_get(func, "loudness_sma3_stddevNorm")
    out["loudness_percentile50"] = safe_get(
        func,
        "loudness_sma3_percentile50.0",
        alt_search=("loudness", "percentile50"),
    )

    # Spectral
    out["alphaRatioV"] = safe_get(func, "alphaRatioV")
    out["alphaRatioUV"] = safe_get(func, "alphaRatioUV")
    out["hammarbergIndexV"] = safe_get(func, "hammarbergIndexV")
    out["hammarbergIndexUV"] = safe_get(func, "hammarbergIndexUV")
    out["slopeV0_500"] = safe_get(func, "slopeV0-500")
    out["slopeV500_1500"] = safe_get(func, "slopeV500-1500")

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
    def add_flag(cond: bool, msg: str):
        if cond:
            flags.append(msg)

    add_flag((out.get("F0_stddevNorm") or 0) >= THRESH["F0_stddevNorm_high"], "F0 변화 폭이 큼")

    # 상승/하강 기울기는 절댓값 기준으로 판단
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
    if any("Jitter" in f or "Shimmer" in f or "HNR" in f for f in flags):
        comments.append("발성 안정성이 낮아 긴장/불안정 음성일 수 있습니다.")
    if any("음량 피크" in f or "발성 구간" in f for f in flags):
        comments.append("큰 소리/짧은 반복 발성 패턴이 관찰됩니다.")

    out["summary_ko"] = " ".join(comments) if comments else "특이 신호가 두드러지지 않습니다 (임계값은 데이터에 맞게 보정 필요)."
    return out

# ===================== 6) 메인 =====================
def main():
    # 1) 영상 → 오디오 (이미 WAV가 있으면 건너뛸 수 있음)
    if not os.path.isfile(AUDIO_PATH):
        extract_wav(FFMPEG_PATH, VIDEO_PATH, AUDIO_PATH, sr=16000, mono=True)
    else:
        print(f"[건너뜀] 오디오가 이미 존재: {AUDIO_PATH}")

    # 2) openSMILE 실행 (Functional + 가능 시 LLD)
    run_opensmile(SMILE_PATH, CONFIG_PATH, AUDIO_PATH, FUNC_CSV, LLD_CSV)

    # 3) 기능 CSV 파싱
    func_row = parse_functionals(FUNC_CSV)

    # 4) LLD CSV 로드(없으면 None)
    lld_df = pd.read_csv(LLD_CSV, sep=";", engine="python").apply(pd.to_numeric, errors='coerce').fillna(0.0)


    # 5) 요약 생성
    summary = build_summary(func_row, lld_df)

    # 6) 저장
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 간단 Markdown 리포트
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
        "---",
        "※ 임계값(THRESH)은 데이터에 맞게 보정하세요. LLD CSV가 없으면 일부 파생 지표는 N/A로 표기됩니다.",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[완료] JSON 리포트: {REPORT_JSON}")
    print(f"[완료] Markdown 리포트: {REPORT_MD}")

    # ===================== 7) 급발진 확률 계산 =====================
    FLAG_WEIGHTS = {
        "F0 변화 폭이 큼": 0.25,
        "F0 상승/하강 기울기 큼": 0.2,
        "Jitter 증가(떨림)": 0.15,
        "Shimmer 증가(음량 미세 변동)": 0.1,
        "HNR 낮음(잡음↑)": 0.1,
        "음량 피크 빈도 증가": 0.1,
        "초당 발성 구간 수 많음": 0.05,
        "발성 구간 평균 길이가 매우 짧음": 0.05,
    }

    def build_sudden_rage_probability(summary: dict) -> dict:
        flags = summary.get("flags", [])
        score = 0.0
        for f in flags:
            score += FLAG_WEIGHTS.get(f, 0.0)
        probability = min(score, 1.0) * 100
        return {
            "sudden_rage_probability_percent": round(probability, 2),
            "active_flags": flags
        }
        # 7) 급발진 확률 계산
    rage_result = build_sudden_rage_probability(summary)
    print("=== 급발진 확률 분석 ===")
    print(f"확률: {rage_result['sudden_rage_probability_percent']}%")
    print("활성 플래그:", rage_result["active_flags"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[에러]", e)
        sys.exit(1)
    
