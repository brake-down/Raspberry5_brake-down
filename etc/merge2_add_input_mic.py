#!/usr/bin/env python3
"""
Real-time pedal misoperation detector (fusion of VIDEO/AUDIO/OBD2)
- 입력: VIDEO 1개 확률값, AUDIO 1개 확률값, OBD2(시리얼) 4개 값 {speed,rpm,throttle,brake}
- 구조: Producer(3) -> Queue -> Consumer(판단 루프) -> 경고/로그
"""

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


# =========================
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
# 3) Producer(입력)
# ===========================

def audio_producer():
    """외부에서 계산된 audio_surprise_prob 1개 값을 받는다고 가정(모의)."""
    period = 1.0 / CFG.audio_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        val = max(0.0, min(1.0, random.gauss(0.15, 0.1)))  # 기본 낮게
        if random.random() < 0.02:
            val = 0.7 + random.random()*0.3
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": val}, src="audio0", seq=seq))
        seq += 1
        time.sleep(period)
        
# === 실제 마이크 기반 오디오 프로듀서 (RealTimeOpenSmile 사용) ===
def _clip01(x): 
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def _to_audio_surprise_prob(m: Dict[str, float]) -> float:
    """
    realtime_analyzer의 지표(loudness, pitch, jitter, shimmer, hnr)를
    0~1 스케일의 audio_surprise_prob로 변환 (경량 휴리스틱).
    필요하면 가중치/스케일을 조정해도 무방.
    """
    # 대략적인 정규화 범위(경험치 기반)
    loud_n   = _clip01(m.get("loudness", 0.0) / 2.0)            # ~2.0 근방까지
    jitter_n = _clip01(m.get("jitter", 0.0) / 0.5)              # 0~0.5
    shimmer_n= _clip01(m.get("shimmer", 0.0) / 3.0)             # 0~3 dB
    # HNR가 낮을수록 (특히 음수) 더 거칠다 → surprise ↑
    hnr      = m.get("hnr", 0.0)
    hnr_bad_n= _clip01((5.0 - hnr) / 10.0)                      # hnr=-5 → 1.0 근처

    # 간단한 가중 평균
    s = (0.45 * loud_n) + (0.2 * jitter_n) + (0.2 * shimmer_n) + (0.15 * hnr_bad_n)

    # 안정성을 위해 살짝 S-curve
    s = _clip01(s)
    s = _clip01(1 / (1 + pow(2.71828, -6*(s - 0.5))))  # sigmoid

    return s

def audio_producer_real(meter: RealTimeOpenSmile, fps: float = 20.0):
    """
    RealTimeOpenSmile에서 최신 지표를 읽어, 사람이 보던 포맷으로 한 줄 출력하고
    Q에 MsgType.AUDIO로 audio_surprise_prob를 넣어준다.
    """
    period = 1.0 / max(1e-6, fps)
    seq = 0
    # 백그라운드 분석 시작
    meter.start()

    # 간단한 EMA로 약간 부드럽게
    ema = None
    alpha = 0.3

    try:
        while not EV_STOP.is_set():
            m = meter.read()  # {"loudness","pitch","jitter","shimmer","hnr"} or None
            if m:
                # 사람이 보는 한 줄 (음성 입력 코드와 동일 포맷)
                print(
                    f"loudness: {m.get('loudness', float('nan')):.2f} | "
                    f"pitch: {m.get('pitch', float('nan')):.2f} | "
                    f"jitter: {m.get('jitter', float('nan')):.2f} | "
                    f"shimmer: {m.get('shimmer', float('nan')):.2f} | "
                    f"hnr: {m.get('hnr', float('nan')):.2f}"
                )

                prob = _to_audio_surprise_prob(m)
                ema = (alpha * prob + (1 - alpha) * ema) if ema is not None else prob
                prob_smoothed = ema

                # 판단 루프가 기대하는 키로 큐에 투입
                Q.put(Msg(
                    MsgType.AUDIO,
                    now_s(),
                    {"audio_surprise_prob": float(prob_smoothed)},
                    src="mic0",
                    seq=seq
                ))
                seq += 1

            time.sleep(period)
    finally:
        meter.stop()


def video_producer():
    """외부에서 계산된 face_surprise_prob 1개 값을 받는다고 가정(모의)."""
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

def serial_producer_real(port="/dev/ttyUSB0", baud=115200):
    """
    보드 → 라즈베리파이 UART로 들어오는 문자열:
      /S<speed>/R<rpm>/P<power>/B<brake>
    예: /S12/R1500/P0.37/B0
    를 읽어 (speed, rpm, throttle, brake)로 큐에 넣는다.
    """
    # P는 정수/소수 모두 허용
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')

    # 마지막 '인정된' 값(노이즈 필터)
    S_val = None  # speed
    R_val = None  # rpm
    P_val = None  # throttle(power)
    B_val = None  # brake

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

                    new_S = int(m.group(1))     # speed
                    new_R = int(m.group(2))     # rpm
                    new_P = float(m.group(3))   # throttle(power)
                    new_B = int(m.group(4))     # brake (0/1)

                    # ===== 노이즈/이상치 필터 =====
                    # S: 직전 값이 양수인데 새 값이 0으로 '뚝' 떨어지면 무시
                    if not (S_val is not None and S_val > 0 and new_S == 0):
                        S_val = new_S
                    # R: 0이면 무시
                    if new_R != 0:
                        R_val = new_R
                    # P: 0.0이면 무시
                    if new_P != 0.0:
                        P_val = new_P
                    # B: 항상 갱신
                    B_val = new_B

                    # 초기화 미완료면 skip
                    if None in (S_val, R_val, P_val, B_val):
                        continue

                    data = {
                        "speed": float(S_val),
                        "rpm": int(R_val),
                        "throttle": float(P_val),
                        "brake": int(B_val),
                    }
                    Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_serial"))
        except Exception as e:
            print(f"[serial] reconnecting due to: {e}")
            time.sleep(1.0)

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
            "brake": 0, "throttle": 0.2, "speed": 30, "rpm": 2000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    # 2단계: WARNING (OBD 이상 + audio peak 순간)
    for _ in range(10):  # 0.5초
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.8}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 10, "rpm": 3500
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    # 3단계: ALERT (OBD 이상 + audio 지속성)
    for _ in range(40):  # 2초
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.9}, seq=seq))
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "brake": 1, "throttle": 0.9, "speed": 5, "rpm": 4000
        }, seq=seq))
        seq += 1
        time.sleep(0.05)

    EV_STOP.set()
    


def serial_producer_loopback_sim(rate_hz: float = 50.0):
    """
    PySerial 가상 포트(loop://)로 '시리얼 쓰기+읽기'를 동시에 수행.
    - writer 스레드가 /S..../R..../P..../B.. 형태의 줄을 loop 포트에 씀
    - 같은 포트를 reader(현재 함수 본문)가 읽어 정규식 파싱 후 큐(Q)에 투입
    => 실제 차 없이도 end-to-end 파서/판단 로직 검증 가능
    """
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    ser = serial_for_url("loop://", baudrate=115200, timeout=1)

    # --- 시나리오를 쓰는 writer 스레드 (정상 → WARNING → ALERT) ---
    def writer():
        dt = 1.0 / rate_hz

        # 1) 정상 3초
        for _ in range(int(3.0 * rate_hz)):
            line = f"/S30/R2000/P0.20/B0\n"
            ser.write(line.encode("utf-8"))
            time.sleep(dt)

        # 2) WARNING 0.5초 (브레이크+고스로틀, 짧은 피크)
        for _ in range(int(0.5 * rate_hz)):
            line = f"/S10/R3500/P0.90/B1\n"
            ser.write(line.encode("utf-8"))
            time.sleep(dt)

        # 3) ALERT 2초 (브레이크+고스로틀 지속)
        for _ in range(int(2.0 * rate_hz)):
            line = f"/S5/R4000/P0.90/B1\n"
            ser.write(line.encode("utf-8"))
            time.sleep(dt)

        # 필요하면 계속 루프 돌리려면 위 구간을 while로 감싸면 됨.
        # 여기선 한 번만 쏘고 종료
        # 끝낼 때 메인 루프가 ALERT를 내도록 충분한 데이터가 들어간 상태.

    threading.Thread(target=writer, daemon=True).start()

    # --- reader: loop://에서 방금 쓴 줄을 읽어 파싱 → 큐로 투입 ---
    # (네가 만든 serial_producer_real의 read/파싱/필터링 부분과 동일 패턴)
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

        # 노이즈/이상치 필터 (네 로직 그대로)
        if not (S_val is not None and S_val > 0 and new_S == 0):
            S_val = new_S
        if new_R != 0:
            R_val = new_R
        if new_P != 0.0:
            P_val = new_P
        B_val = new_B

        if None in (S_val, R_val, P_val, B_val):
            continue

        data = {
            "speed": float(S_val),
            "rpm": int(R_val),
            "throttle": float(P_val),
            "brake": int(B_val),
        }
        Q.put(Msg(MsgType.SERIAL, now_s(), data, src="obd_loop"))


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
    # ★ 마이크 분석기 준비
    meter = RealTimeOpenSmile()    
    
    ths = [
        #threading.Thread(target=audio_producer, daemon=True),
        threading.Thread(target=audio_producer_real,kwargs={"meter": meter, "fps": CFG.audio_hz},daemon=True),
        threading.Thread(target=video_producer, daemon=True),

         # 🔁 실제 차량 대신 가상 시리얼(loop://) 시뮬레이터 사용
        threading.Thread(target=serial_producer_loopback_sim, kwargs={"rate_hz": 50.0}, daemon=True),


        # # ★ 실제 시리얼 리더 사용 (입력 형식: {"speed","rpm","throttle","brake"})
        # threading.Thread(
        #     target=serial_producer_real,
        #     kwargs={"port": "/dev/ttyUSB0", "baud": 115200},
        #     daemon=True
        # ),

        threading.Thread(target=decision_loop, daemon=True),
    ]

    # === 테스트 시나리오만 돌릴 땐 아래로 교체 ===
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
        try:
            meter.stop()
        except Exception:
            pass
        for t in ths:
            t.join(timeout=1.0)
        print("Stopped.")

if __name__ == "__main__":
    main()
