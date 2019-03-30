import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

filename = "hide.wav"
rate, data = scipy.io.wavfile.read(filename)
time = np.linspace(0, len(data) / rate, num=len(data))

plt.subplot(2, 1, 1)
plt.title("Original")
plt.plot(time, data)
plt.xlabel('time (seconds)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.title("New")
plt.plot(time, data / 32768)
plt.xlabel('time (seconds)')
plt.ylabel('Amplitude')

plt.show()
