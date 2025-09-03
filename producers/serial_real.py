# producers/serial_real.py
import time, re, serial
from fusion import Msg, MsgType, now_s

def serial_producer_real(Q, port="/dev/ttyUSB0", baud=115200):
    pattern = re.compile(r'/S(\d+)/R(\d+)/P([0-9]+(?:\.[0-9]+)?)/B(\d+)')
    S_val = R_val = P_val = B_val = None
    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                try: ser.reset_input_buffer()
                except: pass
                while True:
                    raw = ser.readline()
                    if not raw: continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    m = pattern.fullmatch(line)
                    if not m: continue
                    new_S, new_R, new_P, new_B = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
                    if not (S_val is not None and S_val > 0 and new_S == 0): S_val = new_S
                    if new_R != 0: R_val = new_R
                    if new_P != 0.0: P_val = new_P
                    B_val = new_B
                    if None in (S_val, R_val, P_val, B_val): continue
                    Q.put(Msg(MsgType.SERIAL, now_s(), {
                        "speed": float(S_val), "rpm": int(R_val), "throttle": float(P_val), "brake": int(B_val)
                    }, src="obd_serial"))
        except Exception as e:
            print(f"[serial] reconnecting due to: {e}")
            time.sleep(1.0)
