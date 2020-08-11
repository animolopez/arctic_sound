import sys
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

sys.path.append(r"C:\Users\作業用\Documents\module")
from iowave import *
from dB import *
from spectrogram import *
from window import *

input_file = r"C:\Users\作業用\Documents\九州大学\JAPANESE_44100\F01\NJ01F001_44100.wav"
segdur = [20, 40, 60, 80, 160] # segment duration [ms]

X = readGetWave(input_file)
frs = X[0][2]
numsamp = X[0][3]
x = X[1]
Lx = Leq(x)
x_length = np.float64(numsamp / frs)
showSpectrogram(x,x_length,N=1024,frs=frs,Window='hamming')

for l in segdur:
    segsamp = int(np.round(l * 0.001 * frs))
    seginfo = divmod(numsamp, segsamp)
    segrep = seginfo[0] + 1
    padsamp = segsamp - seginfo[1]

    x_zeropad = np.pad(x, [0,padsamp], 'constant')
    x_matrix = x_zeropad.reshape([segrep, segsamp])
    x_rev = np.fliplr(x_matrix)
    env = sqrtCosEnv(l*0.001,rft=0.005,frs=frs)
    y_matrix = x_rev * env
    y = np.ravel(y_matrix)

    Ly = Leq(y)
    y_cal = amplify((Lx - Ly), y)
    y_length = np.float64(y_cal.size / frs)
    showSpectrogram(y_cal,y_length,N=1024,frs=frs,Window='hamming')

    IPython.display.Audio(y_cal, rate = frs)

    output_file = r"C:\Users\作業用\Documents\python\NJ01F001_44100_ltr_%d.wav" % l
    writeWave(output_file, y_cal, params=(1, 2, 44100))
