#!/usr/bin/env python3
"""
Real-time pedal misoperation detector (fusion of VIDEO/AUDIO/OBD2)
- 입력: VIDEO 1개 확률값, AUDIO 1개 확률값, OBD2 다중 값
- 구조: Producer(3) -> Queue -> Consumer(판단 루프) -> 경고/로그
"""

import time
import json
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

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
    neutral_throttle_thr: float = 0.4          # gear=='N' & throttle>0.4
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

# ===========================
# 3) Producer(입력) - 모의버전
# ===========================

def audio_producer():
    """외부에서 계산된 audio_surprise_prob 1개 값을 받는다고 가정.
    지금은 모의로 0~1 사이 값을 생성."""
    period = 1.0 / CFG.audio_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        # TODO: 외부 입력으로 교체 (소켓/파이프/SDK 등)
        val = max(0.0, min(1.0, random.gauss(0.15, 0.1)))  # 기본 낮게
        # 가끔 놀람 상황 모의
        if random.random() < 0.02:
            val = 0.7 + random.random()*0.3
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": val}, src="audio0", seq=seq))
        seq += 1
        time.sleep(period)

def video_producer():
    """외부에서 계산된 face_surprise_prob 1개 값을 받는다고 가정.
    지금은 모의로 0~1 사이 값을 생성."""
    period = 1.0 / CFG.video_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        # TODO: 외부 입력으로 교체
        val = max(0.0, min(1.0, random.gauss(0.12, 0.1)))
        # 가끔 놀람 상황 모의
        if random.random() < 0.015:
            val = 0.65 + random.random()*0.35
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": val}, src="cam0", seq=seq))
        seq += 1
        time.sleep(period)

def serial_producer():
    """OBD2 시리얼 파싱 결과를 받는다고 가정. 지금은 모의 값 생성."""
    period = 1.0 / CFG.obd_hz
    seq = 0
    import random
    speed = 10.0
    rpm = 1200.0
    throttle = 0.1
    brake = 0
    gear = "D"

    while not EV_STOP.is_set():
        # TODO: 실제 OBD2 파서로 교체
        # 평상시 변화(완만)
        throttle = max(0.0, min(1.0, throttle + random.uniform(-0.02, 0.02)))
        speed = max(0.0, speed + throttle*0.5 - brake*0.7 + random.uniform(-0.2, 0.2))
        rpm = max(600.0, rpm + (throttle*800 - brake*500) + random.uniform(-50, 50))

        # 가끔 비정상(브레이크+스로틀 동시) 모의
        conflict = False
        if random.random() < 0.01:
            brake = 1
            throttle = 0.85
            rpm += 800
            speed += random.uniform(-0.5, 0.5)
            conflict = True
        else:
            brake = 1 if random.random() < 0.05 else 0

        data = {
            "brake": int(brake),
            "throttle": float(throttle),
            "speed": float(speed),
            "gear": gear,
            "rpm": int(rpm),
            "conflict": bool(conflict),  # 외부 파서가 줄 수도 있는 플래그
        }
        Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd0", seq=seq))
        seq += 1
        time.sleep(period)

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
        # 오래된 것은 버림
        prune_buffer(self.audio, self.horizon)
        prune_buffer(self.video, self.horizon)
        prune_buffer(self.serial, self.horizon)

def is_obd_abnormal(state: FusionState) -> (bool, List[str]):
    """OBD 비정상 후보 규칙들 체크."""
    why = []
    if not state.serial:
        return False, why
    cur = state.serial[-1].data

    brake = int(cur.get("brake", 0))
    throttle = float(cur.get("throttle", 0.0))
    speed = float(cur.get("speed", 0.0))
    gear = str(cur.get("gear", "D"))
    # 최근 0.5초 변화량
    drpm = delta_over_window(state.serial, "rpm", CFG.recent_window_sec)
    dspeed = delta_over_window(state.serial, "speed", CFG.recent_window_sec)

    # 규칙들
    cond1 = (brake == 1 and throttle > CFG.throttle_brake_conflict_thr)
    cond2 = (throttle > CFG.throttle_high_thr and speed < CFG.low_speed_thr)
    cond3 = (gear == "N" and throttle > CFG.neutral_throttle_thr)
    cond4 = (drpm > CFG.rpm_jump_thr and abs(dspeed) < CFG.speed_small_thr)

    if cond1: why.append("OBD:brake+throttle_conflict")
    if cond2: why.append("OBD:high_throttle_low_speed")
    if cond3: why.append("OBD:neutral_accel")
    if cond4: why.append("OBD:rpm_jump_speed_static")

    return (cond1 or cond2 or cond3 or cond4), why

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
            "gear": cur_obd.get("gear"),
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
    """
    1) 정상 상황 (OK)
    2) WARNING 상황 (OBD 이상 + audio peak 순간)
    3) ALERT 상황 (OBD 이상 + audio 지속성)
    순서대로 데이터를 밀어넣는 시뮬레이터
    """
    seq = 0

    # 1단계: 정상 (3초간 OK)
    for _ in range(60):  # 20Hz * 3s
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.1}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.1}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 0, "throttle": 0.2, "speed": 30, "gear": "D", "rpm": 2000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    # 2단계: WARNING (OBD 이상 + audio peak 순간)
    for _ in range(10):  # 0.5초
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.8}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 10, "gear": "D", "rpm": 3500
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    # 3단계: ALERT (OBD 이상 + audio 지속성)
    for _ in range(40):  # 2초 (0.05s*40=2s)
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.9}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 5, "gear": "D", "rpm": 4000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    # 종료 이벤트 발생
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

        # 주기적으로 판단 (20Hz 근처)
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
        threading.Thread(target=audio_producer, daemon=True),
        threading.Thread(target=video_producer, daemon=True),
        threading.Thread(target=serial_producer, daemon=True),
        threading.Thread(target=decision_loop, daemon=True),
    ]
    
    # === 테스트 시나리오 (정상→WARNING→ALERT) ===
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
