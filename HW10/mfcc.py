import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa as lb

filename = "hello.wav"
rate, data = scipy.io.wavfile.read(filename)
signal = data / 32768

mfcc = lb.feature.mfcc(signal, rate, n_mfcc=13, n_fft=int(rate * 0.025), hop_length=int(rate * 0.01))
mfcc_delta = lb.feature.delta(mfcc)
double_mfcc_delta = lb.feature.delta(mfcc_delta)


plt.subplot(3, 1, 1)
plt.title("MFCC")
plt.plot(np.transpose(mfcc[0:3]), linewidth=1)
plt.xlabel('Frame index')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 2)
plt.title("Delta MFCC")
plt.plot(np.transpose(mfcc_delta[0:3]), linewidth=1)
plt.xlabel('Frame index')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 3)
plt.title("Double Delta MFCC")
plt.plot(np.transpose(double_mfcc_delta[0:3]), linewidth=1)
plt.xlabel('Frame index')
plt.ylabel('Amplitude')
plt.grid()

plt.subplots_adjust(wspace=0, hspace=1)
plt.show()

