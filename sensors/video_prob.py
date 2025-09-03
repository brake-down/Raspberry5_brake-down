# sensors/video_prob.py
import os
import time
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
import tensorflow as tf

# 얼굴 검출기 (OpenCV haarcascade)
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# FER 라벨 순서 (모델에 맞게 조정)
EXPR_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def init_video_model(model_path: str):
    """
    TFLite 모델 로드 / 텐서 정보 반환
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"FER TFLite model not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # 입력 텐서 크기에서 H, W 추론
    ishape = input_details[0]['shape']  # [1, H, W, C]
    h, w = int(ishape[1]), int(ishape[2])
    # 채널은 대부분 1(그레이) 혹은 3(RGB) 모델. 여기선 1을 가정(질문 코드 기준).
    return interpreter, input_details, output_details, (w, h)

def _preprocess_face_roi(gray_face: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """그레이 얼굴 ROI를 모델 입력 크기로 리사이즈 + [0,1] 스케일 + (1,H,W,1) 배치화"""
    w, h = size_wh
    face_resized = cv2.resize(gray_face, (w, h), interpolation=cv2.INTER_AREA)
    face_resized = np.expand_dims(face_resized, axis=-1)  # (H,W,1)
    input_data = np.expand_dims(face_resized, axis=0).astype(np.float32)  # (1,H,W,1)
    input_data = input_data / 255.0
    return input_data

def analyze_frame_for_emergency(
    frame_bgr: np.ndarray,
    interpreter,
    input_details,
    output_details,
    size_wh: Tuple[int, int] = (64, 64),
) -> Dict[str, Any]:
    """
    한 프레임에서 얼굴 검출→감정 추론→'급발진 확률' 산출
    반환: dict (항상 리턴, 얼굴 미검출이어도 face_detected=False 로 기록)
    """
    ts = time.time()

    # 1) 얼굴 검출
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    result: Dict[str, Any] = {
        "ts": ts,
        "face_detected": False,
        "expr": "NoFace",
        "expr_index": None,
        "raw": {},
        "final_prob": 0.0,        # 0~1
        "final_prob_pct": 0.0,    # 0~100
        "face_box": None,         # (x,y,w,h) or None
    }

    if len(faces) == 0:
        return result  # 얼굴 없으면 확률 0으로 리턴

    # 가장 큰 얼굴 하나 사용 (여럿이면 첫 번째로 충분)
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    # 2) 전처리 & 추론
    input_tensor = _preprocess_face_roi(face_roi, size_wh)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # (7,) 예상

    # 3) 레이블/확률
    expr_idx = int(np.argmax(output))
    expr_name = EXPR_LABELS[expr_idx]
    probs = {EXPR_LABELS[i]: float(output[i]) for i in range(len(EXPR_LABELS))}

    # 4) 급발진 확률 조합식 (질문 코드 기반, 0~1로 클램핑)
    surprise = probs.get('Surprise', 0.0)
    fear = probs.get('Fear', 0.0)
    angry = probs.get('Angry', 0.0)
    neutral = probs.get('Neutral', 0.0)
    happy = probs.get('Happy', 0.0)

    raw_score = (surprise * 0.90) + (fear * 0.7) + (angry * 0.4) - ((neutral + happy) * 0.3)
    final_prob = min(1.0, max(0.0, raw_score))
    final_prob_pct = round(final_prob * 100.0, 2)

    result.update({
        "face_detected": True,
        "expr": expr_name,
        "expr_index": expr_idx,
        "raw": probs,
        "final_prob": float(final_prob),
        "final_prob_pct": float(final_prob_pct),
        "face_box": (int(x), int(y), int(w), int(h)),
    })
    return result

def draw_overlay(frame_bgr: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """
    미리보기용 오버레이 (얼굴 박스/라벨/확률). 원본을 수정해서 반환.
    """
    if analysis.get("face_detected") and analysis.get("face_box"):
        (x, y, w, h) = analysis["face_box"]
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    text = f"{analysis.get('expr','-')} | {analysis.get('final_prob_pct',0):.1f}%"
    cv2.putText(frame_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 220, 10), 2, cv2.LINE_AA)
    return frame_bgr
