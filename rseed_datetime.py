# Fix np.random.seed by datetime to get reproducibility of stimuli

import numpy as np
import datetime
import os

# Get current time with class 'int'
dt_now = datetime.datetime.now()
print(dt_now)
dt_str = datetime.datetime.strftime(dt_now, '%Y/%m/%d %H:%M')
print(dt_str)
time_str = dt_now.strftime('%H%M') # get current time (HHMM, class 'str')
print(time_str)
print(type(time_str))
time_int = int(time_str) # current time (HHMM, class 'int')
print(time_int)
print(type(time_int))

# Record and export logs of a random seed
output_dir = 'C:\\Users\\hikar\\Documents\\python\\test\\'
logfile_name = 'timelog.txt'
#os.chdir(output_dir)
#print(os.getcwd())
path = output_dir + logfile_name
path
if os.path.isfile(path) is True:
    with open(path, mode='a') as log:
        log.write('\nThe random seed is: %s' % time_str)
else:
    with open(path, mode='w') as log:
        log.write('The random seed is: %s' % time_str)

# Fix np.random.seed by datetime
np.random.seed(seed=time_int)
np.random.rand()
