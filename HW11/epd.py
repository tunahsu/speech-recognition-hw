import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math
from dspBox import frameMat

def epd(volume, time):
    threshold = 1.5
    volume = volume.tolist()

    for idx, vo in enumerate(volume):
        if vo >= threshold:
            start = idx * (time / len(volume))
            break
    for idx, vo in enumerate(reversed(volume)):
        if vo >= threshold:
            end = (len(volume) - idx) * (time / len(volume))
            break
    return start, end

def show(filename, scale, frameSize, overlap):
    rate, data = scipy.io.wavfile.read(filename)
    time = np.linspace(0, len(data) / rate, num=len(data))

    enframe = frameMat(data, frameSize, overlap)
    enframe = np.transpose(enframe)

    abs_volume = []
    for frame in enframe:
        abs_volume.append(sum(abs((frame - np.mean(frame)) / scale)))
    abs_volume = np.array(abs_volume)

    start, end = epd(abs_volume, np.size(data) / rate)

    plt.subplot(1, 1, 1)
    plt.plot(time, (data - np.mean(data)), linewidth=1)
    plt.plot([start, start], [-3000, 3000], 'r', linewidth=1)
    plt.plot([end, end], [-3000, 3000], 'r', linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    plt.show()

show('hello.wav', 32768, 512, 128)

