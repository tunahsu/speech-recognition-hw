import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

def show(filename, fr, to):
    rate, data = scipy.io.wavfile.read(filename)
    time = np.linspace(0, len(data) / rate, num=len(data))

    index1 = fr / rate
    index2 = to / rate

    plt.subplot(2, 1, 1)
    plt.title(filename)
    plt.plot(time, (data - np.mean(data)))
    plt.plot([index1, index1], [-32768, 32767], 'g', linewidth=1)
    plt.plot([index2, index2], [-32768, 32767], 'g', linewidth=1)
    plt.xlabel('time (seconds)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(to - fr), (data - np.mean(data))[fr:to])
    plt.xlabel('framesize = %s' % (to - fr))
    plt.ylabel('Amplitude')

    plt.show()

show('a.wav', 2000, 2512)
show('p.wav', 2000, 2512)
