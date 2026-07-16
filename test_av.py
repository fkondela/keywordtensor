import av
import numpy as np

y = np.zeros((1, 4800), dtype=np.float32)
frame = av.AudioFrame.from_ndarray(y, format='flt', layout='mono')
frame.sample_rate = 48000

resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)
resampled_frames = resampler.resample(frame)
for f in resampled_frames:
    print(f.to_ndarray().shape)
