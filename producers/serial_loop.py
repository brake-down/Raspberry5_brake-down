import time, re, threading
from serial import serial_for_url
from fusion import Msg, MsgType, now_s

def serial_producer_loopback_sim(Q, rate_hz: float = 50.0, stop_event=None):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    ser = serial_for_url("loop://", baudrate=115200, timeout=1)

    # stop_event 미지정 시 무한루프용 더미
    class _Dummy:
        def is_set(self): return False
    if stop_event is None:
        stop_event = _Dummy()

    def writer():
        dt = 1.0 / rate_hz
        # 정상 3초
        for _ in range(int(3.0 * rate_hz)):
            if stop_event.is_set(): break
            ser.write(b"/S30/R2000/P0.20/B0\n"); time.sleep(dt)
        # WARNING 0.5초
        for _ in range(int(0.5 * rate_hz)):
            if stop_event.is_set(): break
            ser.write(b"/S10/R3500/P0.90/B1\n"); time.sleep(dt)
        # ALERT 2초
        for _ in range(int(2.0 * rate_hz)):
            if stop_event.is_set(): break
            ser.write(b"/S5/R4000/P0.90/B1\n"); time.sleep(dt)

    threading.Thread(target=writer, daemon=True).start()

    S_val = R_val = P_val = B_val = None
    while not stop_event.is_set():
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        m = pattern.fullmatch(line)
        if not m:
            continue

        new_S = int(m.group(1)); new_R = int(m.group(2))
        new_P = float(m.group(3)); new_B = int(m.group(4))

        if not (S_val is not None and S_val > 0 and new_S == 0): S_val = new_S
        if new_R != 0: R_val = new_R
        if new_P != 0.0: P_val = new_P
        B_val = new_B

        if None in (S_val, R_val, P_val, B_val):
            continue

        Q.put(Msg(MsgType.SERIAL, now_s(), {
            "speed": float(S_val), "rpm": int(R_val),
            "throttle": float(P_val), "brake": int(B_val)
        }, src="obd_loop"))
