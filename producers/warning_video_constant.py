# producers/warning_video_constant.py
import time, random
from typing import Optional
from fusion import Msg, MsgType, now_s
from utils.qput import q_put_drop_oldest

def clamp01(x): 
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def warning_video_constant(
    Q, hz: float = 10.0, stop_event: Optional[object] = None,
    level: float = 0.85, jitter: float = 0.0, src: str = "cam_warn_const"
):
    """
    사람 채널을 항상 높은 값으로 내보내는 상수 시뮬레이터.
    - level: 기본값 (0~1)
    - jitter: ±jitter 범위로 약간 흔들리게 (0이면 고정)
    """
    dt = 1.0 / max(1e-9, hz)
    seq = 0
    while not (stop_event and stop_event.is_set()):
        v = level + (random.random() - 0.5) * 2.0 * jitter
        v = clamp01(v)
        msg = Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": v}, src=src, seq=seq)
        q_put_drop_oldest(Q, msg)
        seq += 1
        time.sleep(dt)
