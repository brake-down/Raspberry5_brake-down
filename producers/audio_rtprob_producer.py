# producers/audio_rtprob_producer.py
import time
from typing import Optional

from sensors.voice_rtprob import VoiceRTProb
from fusion import Msg, MsgType, now_s, CFG
from utils.qput import q_put_drop_oldest

__all__ = ["audio_rtprob_producer"]

def audio_rtprob_producer(
    Q,
    stop_event,
    meter: Optional[VoiceRTProb] = None,
    hz: Optional[float] = None,
    src: str = "mic0",
):
    """
    VoiceRTProb를 사용해 audio_surprise_prob(0~1)을 Q로 주기적으로 푸시.
    - Q          : queue.Queue[Msg]
    - stop_event : threading.Event
    - meter      : (선택) 외부에서 만든 VoiceRTProb 인스턴스 (공유/재사용)
    - hz         : 송출 주기(기본 CFG.audio_hz)
    - src        : 메시지 src 태그
    """
    local_meter = meter or VoiceRTProb()
    started_here = False

    if not getattr(local_meter, "_running", False):
        local_meter.start()
        started_here = True

    period = 1.0 / float(hz or CFG.audio_hz)
    seq = 0
    dropped = 0
    next_t = time.monotonic()

    try:
        while not stop_event.is_set():
            # 1) 실시간 확률 읽기
            prob = float(local_meter.get_prob(0.0))  # 0~1
            prob = 0.0 if prob < 0.0 else 1.0 if prob > 1.0 else prob

            # 2) 메시지 생성 & 비차단 큐 삽입 (가득 차면 가장 오래된 것 드랍)
            msg = Msg(
                MsgType.AUDIO,
                now_s(),
                {"audio_surprise_prob": prob},
                src=src,
                seq=seq,
            )
            if q_put_drop_oldest(Q, msg):
                dropped += 1
            seq += 1

            # 3) 드리프트 없는 주기 스케줄링
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # 심하게 밀렸으면 스케줄 재정렬
                if delay < -period:
                    next_t = time.monotonic()
    finally:
        if started_here:
            local_meter.stop()
        if dropped:
            print(f"[audio_rtprob] dropped={dropped}", flush=True)
