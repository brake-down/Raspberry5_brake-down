# producers/serial_loop.py
import re
import time
import threading
from typing import Optional

from serial import serial_for_url
from fusion import Msg, MsgType, now_s
from utils.qput import q_put_drop_oldest


def serial_producer_loopback_sim(
    Q,
    rate_hz: float = 50.0,
    stop_event: Optional[threading.Event] = None,
    src: str = "obd_loop",
):
    """
    loop:// 가상 포트로 '시리얼 쓰기+읽기' 동시 수행 (무한 반복 시나리오)
    - writer 스레드: 정상(3s) → WARNING(0.5s) → ALERT(2s) → idle(1s) 를 계속 반복
    - reader 루프: 같은 포트에서 라인 읽어 파싱 → 큐(Q)로 투입
    """
    ser = serial_for_url("loop://", baudrate=115200, timeout=1)
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')

    # --- writer: 무한 반복 ---
    def writer():
        # dt = 1.0 / max(1e-6, rate_hz)
        # while not (stop_event and stop_event.is_set()):
        #     # 1) 정상 3초
        #     for _ in range(int(3.0 * rate_hz)):
        #         ser.write(b"/S30/R2000/P0.20/B0\n")
        #         time.sleep(dt)
        #     # 2) WARNING 0.5초
        #     for _ in range(int(0.5 * rate_hz)):
        #         ser.write(b"/S10/R3500/P0.90/B1\n")
        #         time.sleep(dt)
        #     # 3) ALERT 2초
        #     for _ in range(int(2.0 * rate_hz)):
        #         ser.write(b"/S5/R4000/P0.90/B1\n")
        #         time.sleep(dt)
        #     # 4) idle 1초 (선택)
        #     for _ in range(int(1.0 * rate_hz)):
        #         ser.write(b"/S0/R800/P0.00/B0\n")
        #         time.sleep(dt)
        dt = 1.0 / max(1e-6, rate_hz)
        while not (stop_event and stop_event.is_set()):
            # 1) 정상 5초 (throttle=0.2, brake=0)
            # for _ in range(int(5.0 * rate_hz)):
            #     ser.write(b"/S30/R2000/P0.20/B0\n")
            #     time.sleep(dt)

            # 2) ALERT 3초 (throttle=0.9, brake=1)
            for _ in range(int(3.0 * rate_hz)):
                ser.write(b"/S10/R3500/P0.90/B1\n")
                time.sleep(dt)

    threading.Thread(target=writer, daemon=True).start()

    # --- reader: loop://에서 쓴 줄을 읽어 파싱 → 큐로 투입 ---
    S_val = R_val = P_val = B_val = None
    dropped = 0
    try:
        while not (stop_event and stop_event.is_set()):
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

            # 이상치/노이즈 필터
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

            msg = Msg(MsgType.SERIAL, now_s(), data, src=src)
            if q_put_drop_oldest(Q, msg):
                dropped += 1

        if dropped:
            print(f"[serial_loop] dropped={dropped}", flush=True)

    finally:
        try:
            ser.close()
        except Exception:
            pass
