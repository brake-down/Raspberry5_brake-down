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
from fusion import FusionState, decide, now_s, Msg, MsgType, CFG
from producers.audio_real import audio_producer_real
from producers.video_sim import video_producer
from producers.serial_loop import serial_producer_loopback_sim
# from producers.serial_real import serial_producer_real  # 실제 보드 쓸 때


# 전역 큐/종료 이벤트
Q: queue.Queue[Msg] = queue.Queue(maxsize=1024)
EV_STOP = threading.Event()

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
        threading.Thread(target=audio_producer_real,kwargs={"Q": Q, "meter": meter, "fps": CFG.audio_hz, "stop_event": EV_STOP},daemon=True),
        threading.Thread(target=video_producer,kwargs={"Q": Q, "hz": CFG.video_hz, "stop_event": EV_STOP},daemon=True),
        threading.Thread(target=serial_producer_loopback_sim,kwargs={"Q": Q, "rate_hz": CFG.obd_hz, "stop_event": EV_STOP},daemon=True),
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
