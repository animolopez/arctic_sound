# LTFSM_program for NTT-AT "Multilingual Speech Database 2002"

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

### Settings
input_dir = 'C:\\Users\\作業用\\Documents\\九州大学\\JAPANESE_44100\\F01\\'  # Input directory where input files exist
output_dir = 'C:\\Users\\作業用\\Documents\\九州大学\\JAPANESE_T_F_Rand_Mos\\test\\'  # Output directory where output files export
talker_suffix = 'NJ01F'  # Talker's information
speech_array = np.arange(29,30,1)  # np.arange(start_num, (end_num - 1), 1)
#speech_array = np.arange(1,201,1)
frs_suffix = '_44100'
wav_suffix = '.wav'
#cutoff = np.array([1600, 3400])
#cutoff = np.array([50, 570, 1600, 3400, 7000])
cutoff = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000])
mos_segdur = [20,40,80,160,320] # Mosaic segment duration [ms] (segdur > 20 [ms])
block = [2,3,4,6,8,16] # A number of mosaic segments in each randomize segment
max_rand_segdur = 320 # A max randomize segment duration [ms]
loop_times = 3
silence = 0.32 # Margin added to the begininng and the end [s]
interval = 1 # Silent interval between the speech samples [s]

for k in speech_array:
    if k < 10:
        speech_suffix = '00' + str(k)
    elif k < 100:
        speech_suffix = '0' + str(k)
    elif k < 1000:
        speech_suffix = str(k)
    else:
        error = 'The speech No. should be invalid!'
    input_file = input_dir + talker_suffix + speech_suffix + frs_suffix + wav_suffix

    ### Read an input file
    X = readGetWave(input_file)
    fs = X[0][2]
    numsamp = X[0][3]
    x = X[1]
    Lx = Leq(x)
    #showSpectrogram(x,N=1024,fs=fs,Window='hamming')
    gapsil = np.zeros(int(silence * fs))
    gapintv = np.zeros(int (interval * fs))

    for l in mos_segdur:
        for m in block:
            if (l * m) <= max_rand_segdur:
                for j in np.arange(0,loop_times,1):
                    output_filename = talker_suffix + speech_suffix + frs_suffix + '_lt-fsm_%dbands_%dms_%dms.wav' % ((cutoff.size - 1), l, (l * m))

                    ### Get current time with class 'int'
                    dt_now = datetime.datetime.now()
                    time_str = dt_now.strftime('%S%f') # get current time (SSssssss, class 'str')
                    time_int = int(time_str) # current time (SSssssss, class 'int')

                    ### Record and export logs of datetime
                    logfile_name = 'rseedlog.txt'
                    path = output_dir + logfile_name

                    if os.path.isfile(path) is True:
                        with open(path, mode='a') as log:
                            log.write('\n%s loop%s, The random seed is: %s' % (output_filename, j, time_str))
                    else:
                        with open(path, mode='w') as log:
                            log.write('%s loop%s, The random seed is: %s' % (output_filename, j, time_str))

                    np.random.seed(seed=time_int) # Fix np.random.seed by datetime

                    ### Zero padding
                    mos_segsamp = int(np.round(l * 0.001 * fs))
                    rand_segsamp = mos_segsamp * m
                    rand_seginfo = divmod(numsamp, rand_segsamp)
                    rand_segrep = rand_seginfo[0] + 1
                    padsamp = rand_segsamp - rand_seginfo[1]
                    x_zeropad = np.pad(x, [0,padsamp], 'constant')

                    ### Genarate a white noise
                    Wnoise = np.random.randn(x_zeropad.size)
                    Wnoise_cal = amplify((Lx - Leq(Wnoise)), Wnoise)
                    env = sqrtCosEnv(l*0.001,rft=0.005,fs=fs)

                    for n in np.arange(0,(cutoff.size-1),1):
                        bandpass = scipy.signal.firwin(numtaps=1025, cutoff=[cutoff[n], cutoff[n+1]], fs=fs, pass_zero=False)

                        #band_speech = scipy.signal.fftconvolve(x, bandpass)
                        band_speech = np.convolve(x_zeropad,bandpass,mode='same')
                        #showSpectrogram(band_speech)
                        band_speech_pow = np.square(band_speech)
                        rand_seg_speech_pow = np.split(band_speech_pow.reshape((rand_segrep*m),mos_segsamp), rand_segrep)
                        for o in np.arange(0,rand_segrep,1):
                            fwd_seg_pow_mean = np.mean(rand_seg_speech_pow[o], axis=1)
                            rand_seg_pow_mean = np.random.permutation(fwd_seg_pow_mean)
                            for p in np.arange(0,m,1):
                                seg_pow_mean_array = np.full(mos_segsamp, rand_seg_pow_mean[p])
                                seg_pow_mean_array = seg_pow_mean_array * env
                                if p == 0:
                                    rand_pow_mean_array = seg_pow_mean_array
                                else:
                                    rand_pow_mean_array = np.concatenate((rand_pow_mean_array, seg_pow_mean_array), 0)
                            if o == 0:
                                tile_speech_pow = rand_pow_mean_array
                            else:
                                tile_speech_pow = np.concatenate((tile_speech_pow, rand_pow_mean_array), 0)

                        #band_noise = scipy.signal.fftconvolve(Wnoise_cal, bandpass)
                        band_noise = np.convolve(Wnoise_cal,bandpass,mode='same')
                        #showSpectrogram(band_noise)
                        band_noise_pow = np.square(band_noise)
                        seg_noise_pow = np.split(band_noise_pow, (rand_segrep*m))
                        for o in np.arange(0,(rand_segrep*m),1):
                            seg_pow_mean = np.mean(seg_noise_pow[o])
                            pow_mean_array = np.full(mos_segsamp, seg_pow_mean)
                            if o == 0:
                                tile_noise_pow = pow_mean_array
                            else:
                                tile_noise_pow = np.concatenate((tile_noise_pow, pow_mean_array), 0)

                        power_rate = tile_speech_pow / tile_noise_pow
                        if n == 0:
                            y = np.zeros(band_speech.size)
                        y += np.sqrt(power_rate) * band_noise
                        #showSpectrogram(y)

                    Ly = Leq(y)
                    y_cal = amplify((Lx - Ly), y)
                    #showSpectrogram(y_cal,N=1024,fs=fs,Window='hamming')

                    if loop_times == 1:
                        Y = np.concatenate((gapsil, y_cal, gapsil))
                    else:
                        if j == 0:
                            Y = np.concatenate((gapsil, y_cal))
                        elif j < (loop_times - 1):
                            Y = np.concatenate((Y, gapintv, y_cal))
                        elif j == (loop_times - 1):
                            Y = np.concatenate((Y, gapintv, y_cal, gapsil))

                    #IPython.display.Audio(y_cal, rate = fs)

                    output_file = output_dir + output_filename
                    writeWave(output_file, Y, params=(1, 2, 44100))
