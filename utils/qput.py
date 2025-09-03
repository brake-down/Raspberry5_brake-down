# utils/qput.py
import queue as _queue

def q_put_drop_oldest(Q: _queue.Queue, item) -> bool:
    """
    비블로킹 put. 꽉 차 있으면 가장 오래된 항목 하나 버리고 다시 put 시도.
    드롭이 발생했으면 True를, 아니면 False를 반환.
    """
    try:
        Q.put_nowait(item)
        return False
    except _queue.Full:
        dropped = False
        try:
            Q.get_nowait()   # oldest drop
            dropped = True
        except _queue.Empty:
            pass
        try:
            Q.put_nowait(item)
        except _queue.Full:
            # 아주 드문 레이스: 또 Full이면 그냥 포기 (다음 루프에서 재시도)
            pass
        return dropped
