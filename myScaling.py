import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

filename = "hide.wav"
rate, data = scipy.io.wavfile.read(filename)

plt.subplot(2, 1, 1)
plt.title("Original")
plt.plot(data)

new = []
for i in range(data.shape[0]):
    new.append(data[i] / 32768)
new = np.array(new)

plt.subplot(2, 1, 2)
plt.title("New")
plt.plot(new)
plt.show()
