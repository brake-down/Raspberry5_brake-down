from __future__ import annotations
import time
import json
import queue
import threading
import re
import serial
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from serial import serial_for_url
from realtime_analyzer import RealTimeOpenSmile 



# =========================d
# 0) 설정(튜닝 가능한 값들)
# =========================

@dataclass
class Config:
    # 버퍼/창sf
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
    # 주기(모의 입력용)
    audio_hz: float = 20.0
    video_hz: float = 20.0
    obd_hz: float = 50.0
    # 로깅
    log_to_file: bool = False
    log_path: str = "events.log"
    
    alert_hold_throttle: float = 0.2   # ALERT 유지 조건 (엑셀 값이 이 값보다 크면 계속 ALERT)


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


# =========================
# 2) 유틸: 버퍼/통계 계산기
# =========================
def now_s() -> float:
    return time.time()

def prune_buffer(buf: List[Msg], horizon: float) -> None:
    """버퍼에서 horizon(초)보다 오래된 데이터는 앞에서 제거."""
    t = now_s()
    while buf and (t - buf[0].ts) > horizon:
        buf.pop(0)

def recent_values(buf: List[Msg], key: str, win: float) -> List[float]:
    """최근 win초의 key 값을 추출."""
    t = now_s()
    return [m.data.get(key, 0.0) for m in buf if (t - m.ts) <= win]

def peak(buf: List[Msg], key: str, win: float) -> float:
    vals = recent_values(buf, key, win)
    return max(vals) if vals else 0.0

def count_over(buf: List[Msg], key: str, thr: float, win: float) -> int:
    vals = recent_values(buf, key, win)
    return sum(1 for v in vals if v > thr)

def delta_over_window(buf: List[Msg], key: str, win: float) -> float:
    """최근 win초의 key 변화량(마지막값 - 처음값)."""
    t = now_s()
    arr = [(m.ts, float(m.data.get(key, 0.0))) for m in buf if (t - m.ts) <= win]
    if len(arr) < 2:
        return 0.0
    arr.sort(key=lambda x: x[0])
    return arr[-1][1] - arr[0][1]


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

def is_obd_abnormal(state: FusionState) -> Tuple[bool, List[str]]:
    """OBD 비정상 후보 규칙들 체크."""
    why = []
    if not state.serial:
        return False, why
    cur = state.serial[-1].data

    brake = int(cur.get("brake", 0))
    throttle = float(cur.get("throttle", 0.0))
    speed = float(cur.get("speed", 0.0))
    # 최근 0.5초 변화량
    drpm = delta_over_window(state.serial, "rpm", CFG.recent_window_sec)
    dspeed = delta_over_window(state.serial, "speed", CFG.recent_window_sec)

    # 규칙들 (gear 제거)
    cond1 = (brake == 1 and throttle > CFG.throttle_brake_conflict_thr)
    cond2 = (throttle > CFG.throttle_high_thr and speed < CFG.low_speed_thr)
    cond4 = (drpm > CFG.rpm_jump_thr and abs(dspeed) < CFG.speed_small_thr)

    if cond1: why.append("OBD:brake+throttle_conflict")
    if cond2: why.append("OBD:high_throttle_low_speed")
    if cond4: why.append("OBD:rpm_jump_speed_static")

    return (cond1 or cond2 or cond4), why

def decide(state: FusionState, last_alert_ts: float) -> Dict[str, Any]:
    """최근 버퍼를 바탕으로 최종 의사결정."""
    decision = "OK"
    reasons: List[str] = []
    score = 0.0

    obd_abn, obd_why = is_obd_abnormal(state)
    if obd_abn:
        reasons.extend(obd_why)
        score += 1.0

    # 사람 반응(최근 0.5초 창)
    A_peak = peak(state.audio, "audio_surprise_prob", CFG.recent_window_sec)
    V_peak = peak(state.video, "face_surprise_prob", CFG.recent_window_sec)
    A_persist = count_over(state.audio, "audio_surprise_prob", CFG.TA, CFG.recent_window_sec)
    V_persist = count_over(state.video, "face_surprise_prob", CFG.TV, CFG.recent_window_sec)
    human_peak = max(A_peak, V_peak)

    # 규칙 (지속성 우선, 그다음 피크)
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

    # 쿨다운: 직전 ALERT 후 곧바로 반복 경고 억제
    cur_obd = state.serial[-1].data if state.serial else {}
    throttle = float(cur_obd.get("throttle", 0.0))

    # 쿨다운 억제 (단, throttle 값이 높을 때는 무시하고 ALERT 유지)
    now = now_s()
    if decision == "ALERT" and (now - last_alert_ts) < CFG.cooldown_sec:
        decision = "OK"
        reasons.append("cooldown_suppressed")

    # 스냅샷
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

