import sys
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import IPython.display

sys.path.append(r"C:\Users\作業用\Documents\module")
from iowave import *
from dB import *
from spectrogram import *
from window import *

# Settings
input_file = r"C:\Users\作業用\Documents\九州大学\JAPANESE_44100\F01\NJ01F001_44100.wav"
output_dir = 'C:\\Users\\作業用\\Documents\\python\\'
#cutoff = np.array([1600, 3400])
#cutoff = np.array([50, 570, 1600, 3400, 7000])
cutoff = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000])
output_filename = 'NJ01F001_44100_nvs_%dbands.wav' % (cutoff.size - 1)

# Get current time with class 'int'
dt_now = datetime.datetime.now()
dt_str = datetime.datetime.strftime(dt_now, '%Y/%m/%d %H:%M:%S')
time_str = dt_now.strftime('%H%M%S') # get current time (HHMMSS, class 'str')
time_int = int(time_str) # current time (HHMMSS, class 'int')

# Record and export logs of datetime
logfile_name = 'timelog.txt'
path = output_dir + logfile_name

if os.path.isfile(path) is True:
    with open(path, mode='a') as log:
        log.write('\n%s, The random seed is: %s' % (output_filename, time_str))
else:
    with open(path, mode='w') as log:
        log.write('%s, The random seed is: %s' % (output_filename, time_str))

np.random.seed(seed=time_int) # Fix np.random.seed by datetime

# Read an input file
X = readGetWave(input_file)
fs = X[0][2]
numsamp = X[0][3]
x = X[1]
Lx = Leq(x)
showSpectrogram(x,N=1024,fs=fs,Window='hamming')

Wnoise = np.random.randn(numsamp)
Wnoise_cal = amplify((Lx - Leq(Wnoise)), Wnoise)
#y = np.zeros(numsamp)

for l in np.arange(0,(cutoff.size-1),1):
    bandpass = scipy.signal.firwin(numtaps=1025, cutoff=[cutoff[l], cutoff[l+1]], fs=fs, pass_zero=False)
    band_speech = scipy.signal.fftconvolve(x, bandpass)
    #band_speech = np.convolve(x,bandpass,mode='same')
    #showSpectrogram(band_speech)
    band_speech_pow = np.square(band_speech)
    #showSpectrogram(band_speech_pow)
    band_speech_pow_ma = np.convolve(band_speech_pow, GaussWin(length=0.010,fs=fs), mode='same')
    #showSpectrogram(band_speech_pow_ma)
    band_noise = scipy.signal.fftconvolve(Wnoise_cal, bandpass)
    #band_noise = np.convolve(Wnoise_cal,bandpass,mode='same')
    #showSpectrogram(band_noise)
    band_noise_pow = np.square(band_noise)
    #showSpectrogram(band_noise_pow)
    band_noise_pow_ma = np.convolve(band_noise_pow, GaussWin(length=0.040,fs=fs), mode='same')
    #showSpectrogram(band_noise_pow_ma)
    power_rate = band_speech_pow_ma / band_noise_pow_ma
    #showSpectrogram(power_rate)
    if l == 0:
        y = np.zeros(band_speech.size)
    y += power_rate * band_noise
    showSpectrogram(y)

Ly = Leq(y)
y_cal = amplify((Lx - Ly), y)
y_length = np.float64(y_cal.size / fs)
showSpectrogram(y_cal,N=1024,fs=fs,Window='hamming')

IPython.display.Audio(y_cal, rate = fs)

output_file = output_dir + output_filename
writeWave(output_file, y_cal, params=(1, 2, 44100))
