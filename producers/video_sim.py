# producers/video_sim.py
import time, random
from fusion import Msg, MsgType, now_s, CFG

def video_producer(Q, hz: float | None = None, stop_event=None):
    if hz is None: hz = CFG.video_hz
    period = 1.0 / hz
    seq = 0
    while True:
        val = max(0.0, min(1.0, random.gauss(0.12, 0.1)))
        if random.random() < 0.015:
            val = 0.65 + random.random()*0.35
        Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": val}, src="cam0", seq=seq))
        seq += 1
        time.sleep(period)
