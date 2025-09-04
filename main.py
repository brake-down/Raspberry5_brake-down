#!/usr/bin/env python3
"""
Real-time pedal misoperation detector (fusion of VIDEO/AUDIO/OBD2)
- 입력: VIDEO 1개 확률값, AUDIO 1개 확률값, OBD2(시리얼) 4개 값 {speed,rpm,throttle,brake}
- 구조: Producer(3) -> Queue -> Consumer(판단 루프) -> 경고/로그
"""
import sys

import time
import json
import queue
import threading
import re
import serial
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from serial import serial_for_url
from realtime_analyzer import RealTimeOpenSmile 
from fusion import FusionState, decide, now_s, Msg, MsgType, CFG
from producers.audio_real import audio_producer_real
from producers.video_sim import video_producer
from producers.serial_loop import serial_producer_loopback_sim
from producers.serial_real import serial_producer_real 

from sensors.voice_rtprob import VoiceRTProb
from producers.audio_rtprob_producer import audio_rtprob_producer
from producers.video_rtprob_producer import video_rtprob_producer


from producers.warning_obd_sim import warning_obd_sim
from producers.warning_video_pulse import warning_video_pulse

import numpy as np
import cv2
import os

from PIL import Image

try:
    # 라인마다 바로 쓰고 내부 버퍼 우회
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# ==================
# 5) 로깅/출력 유틸
# ==================

_last_print_prob = None
def emit_event(ev):
    global _last_print_prob
    p = ev.get("score", 0.0)
    if (_last_print_prob is None) or (abs(p - _last_print_prob) >= 0.05) or (ev["decision"] == "ALERT"):
        line = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write(line + "\n"); sys.stdout.flush()
        _last_print_prob = p


# 전역 큐/종료 이벤트
Q: queue.Queue[Msg] = queue.Queue(maxsize=1024)
EV_STOP = threading.Event()

# ===========================
# 3) Producer(입력)
# ===========================

def audio_producer():
    """외부에서 계산된 audio_surprise_prob 1개 값을 받는다고 가정(모의)."""
    period = 1.0 / CFG.audio_hz
    seq = 0
    import random
    while not EV_STOP.is_set():
        val = max(0.0, min(1.0, random.gauss(0.15, 0.1)))  # 기본 낮게
        if random.random() < 0.02:
            val = 0.7 + random.random()*0.3
        Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": val}, src="audio0", seq=seq))
        seq += 1
        time.sleep(period)
        




# ===========================
# 테스트 시뮬레이터 (정상→WARNING→ALERT)
# ===========================

def test_scenario_producer():
    """
    1) 정상 상황 (OK)
    2) WARNING 상황 (OBD 이상 + audio peak 순간)
    3) ALERT 상황 (OBD 이상 + audio 지속성)
    순서대로 데이터를 밀어넣는 시뮬레이터
    """
    seq = 0

    while not EV_STOP.is_set():
        # 1) 정상 3초
        for _ in range(60):  # 20Hz * 3s
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.1}, seq=seq))
            Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.1}, seq=seq))
            Q.put(Msg(MsgType.SERIAL, now_s(), {"brake": 0, "throttle": 0.2, "speed": 30, "rpm": 2000}, seq=seq))
            seq += 1
            time.sleep(0.05)

        # 2) WARNING 0.5초
        for _ in range(10):
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.8}, seq=seq))
            Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
            Q.put(Msg(MsgType.SERIAL, now_s(), {"brake": 1, "throttle": 0.9, "speed": 10, "rpm": 3500}, seq=seq))
            seq += 1
            time.sleep(0.05)

        # 3) ALERT 2초
        for _ in range(40):
            Q.put(Msg(MsgType.AUDIO, now_s(), {"audio_surprise_prob": 0.9}, seq=seq))
            Q.put(Msg(MsgType.VIDEO, now_s(), {"face_surprise_prob": 0.2}, seq=seq))
            Q.put(Msg(MsgType.SERIAL, now_s(), {"brake": 1, "throttle": 0.9, "speed": 5, "rpm": 4000}, seq=seq))
            seq += 1
            time.sleep(0.05)

    # EV_STOP.set() 제거!





# =====================
# 6) 메인 루프/런처
# =====================
screen_w, screen_h = 1024, 600
img_a = cv2.imread("a.png")
img_b = cv2.imread("b.png")

def make_rgb565(img):
    """BGR888 이미지를 RGB565 바이트 배열로 변환"""
    canvas_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = (canvas_rgb[:, :, 0] >> 3).astype(np.uint16)
    g = (canvas_rgb[:, :, 1] >> 2).astype(np.uint16)
    b = (canvas_rgb[:, :, 2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.astype(np.uint16).tobytes()

def show_status(decision: str):
    """상황에 맞는 화면 출력"""
    if decision == "ALERT":
        # 빨간 배경 생성
        red_bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        red_bg[:] = (0, 0, 255)  # BGR → 빨간색

        # b.png를 화면 크기에 맞게 리사이즈
        overlay = cv2.resize(img_b, (screen_w, screen_h))

        # 알파 블렌딩 (투명도 조절해서 b.png가 배경 위에 표시되도록)
        alpha = 0.8  # 0~1 사이 값 (1이면 b.png만 보이고, 0이면 배경만 보임)
        base = cv2.addWeighted(overlay, alpha, red_bg, 1 - alpha, 0)

    else:
        # a.png 전체 화면
        base = cv2.resize(img_a, (screen_w, screen_h))
        if decision == "WARNING":
            # 노란 글씨로 WARNING 표시
            cv2.putText(
                base, "WARNING", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA
            )

    # 프레임버퍼에 쓰기
    fb_img = make_rgb565(base)
    with open("/dev/fb0", "wb") as f:
        f.write(fb_img)


    
def decision_loop():
    state = FusionState(horizon=CFG.buffer_sec)
    last_alert_ts = -1e9

    decide_hz = 20.0
    decide_dt = 1.0 / decide_hz
    next_t = time.monotonic()

    while not EV_STOP.is_set():
        # 1) 큐 비우기 (burst drain)
        drained = 0
        while True:
            try:
                m: Msg = Q.get_nowait()
            except queue.Empty:
                break
            state.add(m)
            drained += 1

        # 2) 고정 주기 판단
        now = time.monotonic()
        if now >= next_t:
            res = decide(state, last_alert_ts)
            if res["decision"] in ("WARNING", "ALERT"):
                if res["decision"] == "ALERT":
                    last_alert_ts = res["ts"]
                emit_event(res)
        
            if res["decision"] == "ALERT":
                last_alert_ts = res["ts"]
            emit_event(res)

            # 상황에 맞게 화면 표시
            show_status(res["decision"])
                
            # 다음 틱 예약 (drift 방지)
            next_t += decide_dt
            if (now - next_t) > decide_dt:   # 심하게 밀려 있으면 스케줄 재정렬
                next_t = now + decide_dt

        # 3) CPU 과점 방지
        time.sleep(0.001)


def main():

    # ★ 마이크 분석기 준비
    meter = VoiceRTProb(debug=False)      
    ths = [
        threading.Thread(
            target=audio_rtprob_producer,
            kwargs={"Q": Q, "stop_event": EV_STOP, "meter": meter, "hz": CFG.audio_hz, "src": "mic0"},
            daemon=True,
        ),
        threading.Thread(
            target=video_rtprob_producer,
            kwargs={"Q": Q, "stop_event": EV_STOP, "hz": CFG.video_hz, "src": "cam0"},
            daemon=True,
        ),
        threading.Thread(
            target=serial_producer_real,
            kwargs={"Q": Q, "port": "/dev/ttyUSB0", "baud": 115200, "stop_event": EV_STOP},
            daemon=True,
        ),
        ##threading.Thread(target=serial_producer_loopback_sim,kwargs={"Q": Q, "rate_hz": CFG.obd_hz, "stop_event": EV_STOP},daemon=True),
        threading.Thread(target=decision_loop, daemon=True),
    ]


    for t in ths: t.start()

    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        EV_STOP.set()
        try:
            meter.stop()
        except Exception:
            pass
        for t in ths:
            t.join(timeout=1.0)
        print("Stopped.")

if __name__ == "__main__":
    main()
