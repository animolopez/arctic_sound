import numpy as np
import matplotlib.pyplot as plt
import IPython.display

import sys
sys.path.append(r"C:\Users\作業用\Documents\module")
from iowave import *
from spectrogram import *

A = 0.8
f = 440
length = 1
frs = 44100

t = np.arange(0, length, 1 / frs)
y = A * np.sin(2*np.pi*f*t)

plt.plot(t, y)
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show()
showSpectrogram(y,length,N=1024,Window='hamming')
showFFT(y,frs=frs,frange=4000)
showSTFT(y,frs=frs,frange=4000)

IPython.display.Audio(y, rate = frs)

writeWave(r"C:\Users\作業用\Documents\python\sinewave.wav", y, params=(1, 2, 44100))
