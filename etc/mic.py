import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 44100   # 샘플레이트 (Hz)
FRAME_DURATION = 0.1  # 0.1초 단위

def get_rms(duration=FRAME_DURATION, samplerate=SAMPLE_RATE, threshold=0.01):
    """마이크 수음을 duration 동안 하고 RMS 값 반환"""
    recording = sd.rec(int(samplerate * duration),
                       samplerate=samplerate,
                       channels=1,
                       dtype='float32')
    sd.wait()
    rms = np.sqrt(np.mean(recording**2))
    return rms

if __name__ == "__main__":
    print("마이크 입력 모니터링 시작 (Ctrl+C로 종료)")
    try:
        while True:
            rms = get_rms()
            print(f"RMS: {rms:.5f}", 
                  "🎤 소리 감지됨" if rms > 0.01 else "…조용함")
            time.sleep(0.01)  # CPU 부하 줄이기 위해 약간의 sleep
    except KeyboardInterrupt:
        print("\n종료합니다.")