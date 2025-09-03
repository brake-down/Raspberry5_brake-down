# producers/audio_real.py
import time
from typing import Dict
from fusion import Msg, MsgType, now_s, CFG

# === 실제 마이크 기반 오디오 프로듀서 (RealTimeOpenSmile 사용) ===
def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def _to_audio_surprise_prob(m: Dict[str, float]) -> float:
    """
    realtime_analyzer 지표(loudness, pitch, jitter, shimmer, hnr)를
    0~1 스케일의 audio_surprise_prob로 변환 (경량 휴리스틱).
    """
    loud_n    = _clip01(m.get("loudness", 0.0) / 2.0)   # ~2.0 근방까지
    jitter_n  = _clip01(m.get("jitter", 0.0) / 0.5)     # 0~0.5
    shimmer_n = _clip01(m.get("shimmer", 0.0) / 3.0)    # 0~3 dB
    hnr       = m.get("hnr", 0.0)
    hnr_bad_n = _clip01((5.0 - hnr) / 10.0)             # hnr=-5 → 1.0 근처

    s = (0.45 * loud_n) + (0.2 * jitter_n) + (0.2 * shimmer_n) + (0.15 * hnr_bad_n)
    # 가벼운 S-curve
    s = _clip01(s)
    s = _clip01(1.0 / (1.0 + (2.71828 ** (-6 * (s - 0.5)))))
    return s

def audio_producer_real(Q, meter, fps: float | None = None, ema_alpha: float = 0.3, stop_event=None):
    """
    meter(read() 제공)에서 최신 지표를 읽어 사람이 보던 포맷을 출력하고,
    Q에 MsgType.AUDIO로 audio_surprise_prob를 넣는다.
    """
    if fps is None:
        fps = CFG.audio_hz
    period = 1.0 / max(1e-6, fps)
    seq = 0

    # stop_event가 없으면 무한 루프(테스트용)
    class _Dummy: 
        def is_set(self) -> bool: return False
    if stop_event is None:
        stop_event = _Dummy()

    meter.start()
    ema = None
    try:
        while not stop_event.is_set():
            m = meter.read()  # {"loudness","pitch","jitter","shimmer","hnr"} or None
            if m:
                # 사람이 보던 한 줄 포맷
                print(
                    f"loudness: {m.get('loudness', float('nan')):.2f} | "
                    f"pitch: {m.get('pitch', float('nan')):.2f} | "
                    f"jitter: {m.get('jitter', float('nan')):.2f} | "
                    f"shimmer: {m.get('shimmer', float('nan')):.2f} | "
                    f"hnr: {m.get('hnr', float('nan')):.2f}"
                )
                prob = _to_audio_surprise_prob(m)
                ema = prob if ema is None else (ema_alpha * prob + (1 - ema_alpha) * ema)

                Q.put(
                    Msg(
                        MsgType.AUDIO,
                        now_s(),
                        {"audio_surprise_prob": float(ema)},
                        src="mic0",
                        seq=seq,
                    )
                )
                seq += 1
            time.sleep(period)
    finally:
        meter.stop()
