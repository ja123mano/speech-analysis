#------------------------------------------------------------------------------------------------------------------
#   Speech data aquisition
#------------------------------------------------------------------------------------------------------------------
import time
import random
import numpy as np
import sounddevice as sd

import pickle
from datetime import datetime

# Experiment configuration
conditions = [('Si', 1), ('No', 2), ('Agua', 3), ('Comida', 4), ('Dormir', 5),('Silencio', 6)]
n_trials = 17

fixation_cross_time = 1
preparation_time = 0.3
training_time = 2
rest_time = 1

trials = n_trials*conditions
random.shuffle(trials)

fs=44100    

# Data aquisition
data = []
for t in trials:

    # Fixation cross
    print ("*********")    
    time.sleep(fixation_cross_time)    
    
    # Preparation time
    print (t[0])
    time.sleep(preparation_time)

    # Task    
    recording = sd.rec(training_time * fs, samplerate=fs, channels=2,dtype='float64')    
    sd.wait()

    data.append((t[0], t[1], recording))

    # Rest time
    print ("----Descansa----")
    time.sleep(rest_time)

# Play records
#for t in data:
#    sd.play(t[2], fs)
#    sd.wait()

# Save data
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
outputFile = open(now + '.obj', 'wb')
pickle.dump(data, outputFile)
outputFile.close()


#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------