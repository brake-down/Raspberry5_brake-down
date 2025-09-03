# producers/warning_video_pulse.py
import time, queue
from fusion import Msg, MsgType, now_s, CFG
from utils.qput import q_put_drop_oldest


def warning_video_pulse(Q, hz=None, stop_event=None, base=0.15, peak=0.85,
                        interval_frames=8, pulse_len_frames=1, src="cam_pulse"):
    if hz is None: hz = CFG.video_hz
    dt = 1.0 / max(1e-6, hz)
    k, dropped = 0, 0
    while not (stop_event and stop_event.is_set()):
        val = peak if (k % interval_frames) < pulse_len_frames else base
        msg = Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": float(val)}, src=src, seq=k)
        if q_put_drop_oldest(Q, msg):
            dropped += 1
        time.sleep(dt)
    if dropped:
        print(f"[video] dropped={dropped}")
