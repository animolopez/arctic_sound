import sys
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import IPython.display

sys.path.append(r"C:\Users\作業用\Documents\Module")
from iowave import *
from dB import *
from spectrogram import *

frs = 44100 # [Hz]
length = 3 # [s]
Lref = 70 # [dB]
samp_length = int(frs * length)
output_dir = 'C:\\Users\\作業用\\Documents\\python\\'
output_filename = 'white_noise_rseed.wav'
output_file = output_dir + output_filename
logfile_name = 'timelog.txt'

# Get current time with class 'int'
dt_now = datetime.datetime.now()
dt_str = datetime.datetime.strftime(dt_now, '%Y/%m/%d %H:%M:%S')
time_str = dt_now.strftime('%H%M%S') # get current time (HHMMSS, class 'str')
time_int = int(time_str) # current time (HHMMSS, class 'int')

# Record and export logs of datetime
#output_dir = 'C:\\Users\\hikar\\Documents\\python\\test\\'
#logfile_name = 'timelog.txt'
#os.chdir(output_dir)
#print(os.getcwd())
path = output_dir + logfile_name
if os.path.isfile(path) is True:
    with open(path, mode='a') as log:
        log.write('\n%s, The random seed is: %s' % (output_filename, time_str))
else:
    with open(path, mode='w') as log:
        log.write('%s, The random seed is: %s' % (output_filename, time_str))

# Fix np.random.seed by datetime
np.random.seed(seed=time_int)

Wnoise = np.random.randn(samp_length)
#temp_max = np.max(np.abs(Wnoise))

Leq_Wnoise = Leq(Wnoise)
y = amplify((Lref - Leq_Wnoise), Wnoise)

print(Leq(y))
print(np.max(np.abs(y)))
showSpectrogram(y,length,N=512,frange=20000,Window='hamming')
showFFT(y)

IPython.display.Audio(y, rate = frs)

writeWave(output_file, y, params=(1, 2, frs))
