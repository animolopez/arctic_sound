# Fix np.random.seed by datetime to get reproducibility of stimuli

import numpy as np
import datetime
import os

# Get current time with class 'int'
dt_now = datetime.datetime.now()
#print(dt_now)
dt_str = datetime.datetime.strftime(dt_now, '%Y/%m/%d %H:%M:%S')
#print(dt_str)
time_str = dt_now.strftime('%H%M%S') # get current time (HHMMSS, class 'str')
#print(time_str)
#print(type(time_str))
time_int = int(time_str) # current time (HHMMSS, class 'int')
#print(time_int)
#print(type(time_int))

# Record and export logs of a random seed
output_dir = 'C:\\Users\\作業用\\Documents\\python\\'
logfile_name = 'timelog.txt'
path = output_dir + logfile_name

if os.path.isfile(path) is True:
    with open(path, mode='a') as log:
        log.write('\nThe random seed is: %s' % time_str)
else:
    with open(path, mode='w') as log:
        log.write('The random seed is: %s' % time_str)

# Fix np.random.seed by datetime
np.random.seed(seed=time_int)
np.random.rand()
