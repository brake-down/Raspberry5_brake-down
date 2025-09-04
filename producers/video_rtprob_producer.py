# producers/video_rtprob_producer.py
import time
import cv2
from picamera2 import Picamera2

from sensors.video_prob_rt_picam import analyze_frame_for_emergency
from fusion import Msg, MsgType, now_s, CFG
from utils.qput import q_put_drop_oldest

__all__ = ["video_rtprob_producer"]

def video_rtprob_producer(Q, stop_event, hz: float = None, src: str = "cam0"):
    """
    Picamera2 + FER 모델 기반으로 face_surprise_prob(0~1)을 Q에 주기적으로 푸시.
    - Q          : queue.Queue[Msg]
    - stop_event : threading.Event
    - hz         : 송출 주기 (기본 CFG.video_hz)
    - src        : 메시지 src 태그
    """
    period = 1.0 / float(hz or CFG.video_hz)
    seq = 0
    dropped = 0
    next_t = time.monotonic()

    # === 카메라 초기화 ===
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # 카메라 워밍업

    try:
        while not stop_event.is_set():
            # 1) 프레임 캡처
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2) 추론 (표정 기반 확률)
            _, prob = analyze_frame_for_emergency(frame_bgr)  # expr_label 버림

            # 3) 확률(0~100%) → 0~1로 변환 후 큐에 넣기
            msg = Msg(
                MsgType.VIDEO,
                now_s(),
                {"face_surprise_prob": float(prob) / 100.0},
                src=src,
                seq=seq,
            )
            if q_put_drop_oldest(Q, msg):
                dropped += 1
            seq += 1

            # 4) FPS 유지
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                if delay < -period:  # 심하게 밀리면 리셋
                    next_t = time.monotonic()
    finally:
        picam2.stop()
        if dropped:
            print(f"[video_rtprob] dropped={dropped}", flush=True)
