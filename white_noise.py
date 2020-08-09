import sys
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

sys.path.append(r"C:\Users\作業用\Documents\python_package")
from iowave import *
from dB import *
from spectrogram import *

frs = 44100 # [Hz]
length = 3 # [s]
Lref = 70 # [dB]
samp_length = int(frs * length)

Wnoise = np.random.randn(samp_length)
#temp_max = np.max(np.abs(Wnoise))

Leq_Wnoise = Leq(Wnoise)
y = amplify((Lref - Leq_Wnoise), Wnoise)

print(Leq(y))
print(np.max(np.abs(y)))
showSpectrogram(y,length,N=512,frange=20000,Window='hamming')

IPython.display.Audio(y, rate = frs)

writeWave(r"C:\Users\作業用\Documents\python\white_noise.wav", y, params=(1, 2, frs))
