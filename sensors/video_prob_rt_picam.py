#!/usr/bin/env python3
"""
Real-time FER + emergency probability (Raspberry Pi 5 + IMX500 + Picamera2)

- 카메라: Picamera2 API 사용
- 모델: fer_model.tflite (TFLite 모델, 64x64 grayscale 입력)
- 출력: 표정 레이블과 급발진 확률 (0~100%)
"""

import sys, os, time, json
import numpy as np
import cv2
import tensorflow as tf
from picamera2 import Picamera2

# ================================
# 모델 및 파라미터
# ================================
MODEL_PATH = "sensors/fer_model.tflite"
SIZE = 64
expression_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 얼굴 검출기 (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# TFLite 모델 로드
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================================
# 분석 함수
# ================================
def analyze_frame_for_emergency(frame):
    """
    입력 프레임(BGR) → 얼굴 검출 → FER 추론 → 급발진 확률 계산
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return "NoFace", 0.0

    # 가장 큰 얼굴만 사용
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y : y + h, x : x + w]

    # 모델 입력 전처리
    face_resized = cv2.resize(face_roi, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    face_norm = np.expand_dims(face_resized, axis=-1)  # (64,64,1)
    input_data = np.expand_dims(face_norm, axis=0).astype(np.float32) / 255.0

    # 추론 실행
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    # 레이블/확률 계산
    expr_idx = np.argmax(prediction)
    expr_label = expression_labels[expr_idx]

    surprise_prob = prediction[expression_labels.index("Surprise")]
    fear_prob = prediction[expression_labels.index("Fear")]
    angry_prob = prediction[expression_labels.index("Angry")]
    neutral_prob = prediction[expression_labels.index("Neutral")]
    happy_prob = prediction[expression_labels.index("Happy")]

    prob = (surprise_prob * 0.9) + (fear_prob * 0.7) + (angry_prob * 0.4) - (
        (neutral_prob + happy_prob) * 0.3
    )
    final_prob = max(0, min(100, prob * 100))

    return expr_label, final_prob


# ================================
# 메인 실행 루프
# ================================
def main(fps=10, show=True):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # 카메라 워밍업

    print("Ready. Press Ctrl+C to stop.")

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            expr, prob = analyze_frame_for_emergency(frame_bgr)

            # 로그(JSON)
            event = {
                "ts": time.time(),
                "expr": expr,
                "prob": round(float(prob), 3),
            }
            print(json.dumps(event, ensure_ascii=False))

            # 시각화
            if show:
                cv2.putText(
                    frame_bgr,
                    f"{expr} ({prob:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("FER-IMX500", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
                    break

            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        picam2.stop()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=10, help="출력 FPS")
    parser.add_argument("--show", action="store_true", help="창에 영상 표시")
    args = parser.parse_args()

    main(fps=args.fps, show=args.show)
