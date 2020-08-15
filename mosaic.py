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
segdur = [20,40,80,160,320] # Segment duration [ms] (segdur > 20 [ms])

# Read an input file
X = readGetWave(input_file)
fs = X[0][2]
numsamp = X[0][3]
x = X[1]
Lx = Leq(x)
showSpectrogram(x,N=1024,fs=fs,Window='hamming')

for n in segdur:
    output_filename = 'NJ01F001_44100_mos_%dbands_%dms.wav' % ((cutoff.size - 1), n)

    # Get current time with class 'int'
    dt_now = datetime.datetime.now()
    time_str = dt_now.strftime('%H%M%S') # get current time (HHMMSS, class 'str')
    time_int = int(time_str) # current time (HHMMSS, class 'int')

    # Record and export logs of datetime
    logfile_name = 'rseedlog.txt'
    path = output_dir + logfile_name

    if os.path.isfile(path) is True:
        with open(path, mode='a') as log:
            log.write('\n%s, The random seed is: %s' % (output_filename, time_str))
    else:
        with open(path, mode='w') as log:
            log.write('%s, The random seed is: %s' % (output_filename, time_str))

    np.random.seed(seed=time_int) # Fix np.random.seed by datetime

    # Zero padding
    segsamp = int(np.round(n * 0.001 * fs))
    seginfo = divmod(numsamp, segsamp)
    segrep = seginfo[0] + 1
    padsamp = segsamp - seginfo[1]
    x_zeropad = np.pad(x, [0,padsamp], 'constant')

    # Genarate a white noise
    Wnoise = np.random.randn(x_zeropad.size)
    Wnoise_cal = amplify((Lx - Leq(Wnoise)), Wnoise)
    #y = np.zeros(numsamp)
    env = sqrtCosEnv(n*0.001,rft=0.005,fs=fs)

    for l in np.arange(0,(cutoff.size-1),1):
        bandpass = scipy.signal.firwin(numtaps=1025, cutoff=[cutoff[l], cutoff[l+1]], fs=fs, pass_zero=False)

        #band_speech = scipy.signal.fftconvolve(x, bandpass)
        band_speech = np.convolve(x_zeropad,bandpass,mode='same')
        #showSpectrogram(band_speech)
        band_speech_pow = np.square(band_speech)
        seg_speech_pow = np.split(band_speech_pow, segrep)
        for m in np.arange(0,segrep,1):
            seg_pow_mean = np.mean(seg_speech_pow[m])
            pow_mean_array = np.full(segsamp, seg_pow_mean)
            pow_mean_array = pow_mean_array * env
            if m == 0:
                tile_speech_pow = pow_mean_array
            else:
                tile_speech_pow = np.concatenate((tile_speech_pow, pow_mean_array), 0)

        #band_noise = scipy.signal.fftconvolve(Wnoise_cal, bandpass)
        band_noise = np.convolve(Wnoise_cal,bandpass,mode='same')
        #showSpectrogram(band_noise)
        band_noise_pow = np.square(band_noise)
        seg_noise_pow = np.split(band_noise_pow, segrep)
        for m in np.arange(0,segrep,1):
            seg_pow_mean = np.mean(seg_noise_pow[m])
            pow_mean_array = np.full(segsamp, seg_pow_mean)
            if m == 0:
                tile_noise_pow = pow_mean_array
            else:
                tile_noise_pow = np.concatenate((tile_noise_pow, pow_mean_array), 0)

        power_rate = tile_speech_pow / tile_noise_pow
        if l == 0:
            y = np.zeros(band_speech.size)
        y += np.sqrt(power_rate) * band_noise
        showSpectrogram(y)

    Ly = Leq(y)
    y_cal = amplify((Lx - Ly), y)
    showSpectrogram(y_cal,N=1024,fs=fs,Window='hamming')

    IPython.display.Audio(y_cal, rate = fs)

    output_file = output_dir + output_filename
    writeWave(output_file, y_cal, params=(1, 2, 44100))
