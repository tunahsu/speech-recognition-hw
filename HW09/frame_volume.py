import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math
from DSPbox import frameMat

def show(filename, scale, frameSize, overlap):
    rate, data = scipy.io.wavfile.read(filename)
    time = np.linspace(0, len(data) / rate, num=len(data))

    enframe, frameCount = frameMat(data, frameSize, overlap)
    enframe = np.transpose(enframe)
    frameTime = (np.linspace(0, frameCount, frameCount) * (512 - 128)) / rate

    abs_volume = []
    log_volume = []
    for frame in enframe:
        abs_volume.append(sum(abs((frame - np.mean(frame)) / scale)))
        log_volume.append(10 * math.log10(sum(((frame - np.mean(frame)) / scale) ** 2)))
    abs_volume = np.array(abs_volume)
    log_volume = np.array(log_volume)

    plt.subplot(3, 1, 1)
    plt.plot(time, (data - np.mean(data)) / scale, linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(frameTime, abs_volume, linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volume (abs sum)')

    plt.subplot(3, 1, 3)
    plt.plot(frameTime, log_volume, linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volume (decibels)')

    plt.subplots_adjust(wspace=0, hspace=0.7)
    plt.show()

show('HappyNewYear.wav', 32768, 512, 128)

