# scripts/video_prob_rt.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import json
import argparse
from typing import Optional

import cv2
import numpy as np

# 상대 임포트
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sensors.video_prob import (
    init_video_model,
    analyze_frame_for_emergency,
    draw_overlay,
)

def open_camera(backend: str, device: str, width: int, height: int, fps: int):
    """
    간단: OpenCV VideoCapture만 사용.
    - backend=v4l2(추천) / auto
    """
    if backend == "auto":
        backend = "v4l2"

    if backend == "v4l2":
        cap = cv2.VideoCapture(device or 0)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open V4L2 device: {device or 0}")
        # 설정 시도
        if width > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0: cap.set(cv2.CAP_PROP_FPS, fps)
        return cap, f"v4l2:{device or 0}"

    raise RuntimeError(f"Unsupported backend: {backend}")

def emit_json(obj, compact: bool = True):
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":") if compact else None)
    try:
        os.write(1, (s + "\n").encode("utf-8"))
    except Exception:
        sys.stdout.write(s + "\n"); sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to fer_model.tflite")
    ap.add_argument("--backend", default="auto", choices=["auto", "v4l2"], help="camera backend")
    ap.add_argument("--device", default="", help="V4L2 device (e.g., /dev/video0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10, help="target processing FPS")
    ap.add_argument("--show", dest="show", action="store_true")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.set_defaults(show=True)
    ap.add_argument("--debug", action="store_true", help="print even when no face")
    ap.add_argument("--min_log_hz", type=float, default=0.0, help="rate-limit console logs (0=every frame)")
    args = ap.parse_args()

    interpreter, input_details, output_details, size_wh = init_video_model(args.model)

    cap, used = open_camera(args.backend, args.device, args.width, args.height, args.fps)
    emit_json({"event": "camera_ready", "backend": used, "size": [args.width, args.height], "fps": args.fps})
    print("Ready. Press Ctrl+C to stop.")

    frame_interval = 1.0 / max(1, args.fps)
    last_log_ts = 0.0
    last_tick = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] no frame")
                time.sleep(0.1)
                continue

            # 분석
            analysis = analyze_frame_for_emergency(frame, interpreter, input_details, output_details, size_wh)

            # 로깅 (얼굴 없을 때도 찍고 싶으면 --debug)
            now = time.monotonic()
            log_ok = True
            if args.min_log_hz > 0:
                if (now - last_log_ts) < (1.0 / args.min_log_hz):
                    log_ok = False

            if log_ok and (analysis["face_detected"] or args.debug):
                emit_json({
                    "ts": analysis["ts"],
                    "expr": analysis["expr"],
                    "face_detected": analysis["face_detected"],
                    "face_surprise_prob": round(analysis["raw"].get("Surprise", 0.0), 3),
                    "final_prob": round(analysis["final_prob"], 3),
                    "final_prob_pct": analysis["final_prob_pct"],
                })
                last_log_ts = now

            # 미리보기
            if args.show:
                draw_overlay(frame, analysis)
                cv2.imshow("IMX500 FER (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # FPS 타겟 맞추기
            now2 = time.monotonic()
            sleep_for = frame_interval - (now2 - last_tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_tick = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == "__main__":
    main()
