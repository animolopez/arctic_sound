import sys
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

sys.path.append(r"C:\Users\作業用\Documents\module")
from iowave import *
from dB import *
from spectrogram import *

input_file = r"C:\Users\作業用\Documents\九州大学\JAPANESE_44100\F01\NJ01F001_44100.wav"
output_file = r"C:\Users\作業用\Documents\python\NJ01F001_44100_reversed.wav"

X = readGetWave(input_file)
frs = X[0][2]
print(frs)
numsamp = X[0][3]
print(numsamp)
x = X[1]
print(x)
Lx = Leq(x)
print(Lx)
print(np.max(np.abs(x)))
length = np.float64(numsamp / frs)
showSpectrogram(x,length,N=1024,frs=frs,Window='hamming')

x_rev = np.flipud(x)
Lx_rev = Leq(x_rev)
print(Lx_rev)
showSpectrogram(x_rev,length,N=1024,frs=frs,Window='hamming')

IPython.display.Audio(x_rev, rate = frs)

writeWave(output_file, x_rev, params=(1, 2, 44100))
