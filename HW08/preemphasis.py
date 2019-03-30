from onesidespectra import One_sided_spectra
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

filename = "a.wav"

rate, data = scipy.io.wavfile.read(filename)
frq, db = One_sided_spectra(data[10000:10512], rate)

preemph_data = np.append(data[0], data[1:] - 0.98 * data[:-1])
frq2, db2 = One_sided_spectra(preemph_data[10000:10512], rate)

scipy.io.wavfile.write('AfHightPass.wav', rate, preemph_data)

plt.subplot(2, 1, 1)
plt.title("Original")
plt.plot(frq, db, linewidth=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound pressure level (dB)')

plt.subplot(2, 1, 2)
plt.title("Emphasized")
plt.plot(frq2, db2, linewidth=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound pressure level (dB)')

plt.subplots_adjust(wspace=0, hspace=0.7)
plt.show()