# producers/warning_obd_sim.py
import time, queue
from fusion import Msg, MsgType, now_s
from utils.qput import q_put_drop_oldest


def warning_obd_sim(Q, rate_hz=50.0, stop_event=None, speed=8.0, rpm=3500, throttle=0.9, brake=1, src="obd_warn"):
    dt = 1.0 / max(1e-6, rate_hz)
    dropped = 0
    while not (stop_event and stop_event.is_set()):
        msg = Msg(MsgType.SERIAL, now_s(), {
            "speed": float(speed), "rpm": int(rpm),
            "throttle": float(throttle), "brake": int(brake)
        }, src=src)
        if q_put_drop_oldest(Q, msg):
            dropped += 1
        time.sleep(dt)
    if dropped:
        print(f"[obd] dropped={dropped}")
