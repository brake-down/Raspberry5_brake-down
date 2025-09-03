import numpy as np, sounddevice as sd
sr=48000
rec = sd.rec(int(sr*1), samplerate=sr, channels=2, dtype='float32')
sd.wait()
print('RMS=', np.sqrt((rec**2).mean(axis=0)))
