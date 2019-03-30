import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

filename = "hello.wav"
rate, data = scipy.io.wavfile.read(filename)

plt.subplot(2, 1, 1)
plt.title("Original")
plt.plot(data)

new = []
for i in range(np.size(data)):
    if data[i] > 0:
        new.append(1 - data[i])
    elif data[i] < 0:
        new.append(-1 - data[i])
    else:
        new.append(data[i])
new = np.array(new)
new = np.flipud(new)

plt.subplot(2, 1, 2)
plt.title("New")
plt.plot(new)
plt.show()
