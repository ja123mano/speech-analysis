import pickle
import numpy as np
from requests import get
#from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd

from python_speech_features import mfcc

# Load audio records
fs = 44100

#file_name = 'D:\\Decodificación del habla\\07_01_2022_14_09_58.obj'
#file_name = 'D:\\Decodificación del habla\\07_04_2022_19_42_44.obj'
file_name = 'D:\\Decodificación del habla\\07_04_2022_19_44_56.obj'
inputFile = open(file_name, 'rb')
data = pickle.load(inputFile)
n_trials = len(data)
inputFile.close()

#------------------------------------------------------------------------------
# Choose track (t) and get the seconds/wave (1/fs)
t = 0
dt = 1/fs
t_size = data[t][2].shape[0]

#------------------------------------------------------------------------------
# Filter signals (between 10 and 15k Hz)
filt = signal.iirfilter(4, [10, 15000], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='ba')

#------------------------------------------------------------------------------
#Double filter for the sound signals
filtered = []
for tr in data:
    ff1 = signal.filtfilt(filt[0], filt[1], tr[2][:,0], method='gust')
    ff2 = signal.filtfilt(filt[0], filt[1], tr[2][:,1], method='gust')
    filtered.append(np.column_stack((ff1, ff2)))

#------------------------------------------------------------------------------
#Def for cutting words
#Cuts from the signal passed the left and right parts of it
#to_center = True, traverses the signal from the extremes to the center, stopping until the values found are >= max value of signal*percentage of each side
#to_center = False, traverses the signal from the center outwards, stopping until the values found are <= max value of the signal*percentage of each side

#Returns a slice of the original signal, cut from the moment a disruption in the intensity was found (depending on percentages given)

def word_cut(data, per_left=0.1, per_right=0.1, to_center=True):
    ch0 = data[t][:,0]
    ch1 = data[t][:,1]
    max_abs_ch0 = np.argmax(ch0)
    max_abs_ch1 = np.argmax(ch1)

    start_ch0, end_ch0 = 0, 0
    start_ch1, end_ch1 = 0, 0

    if to_center:
        for i in range(len(ch0)):
            if abs(ch0[i]) >= ch0[max_abs_ch0]*per_left:
                start_ch0 = i
                break

        for j in range(len(ch0)-1, 0, -1):
            if abs(ch0[j]) >= ch0[max_abs_ch0]*per_right:
                end_ch0 = j
                break

        for i in range(len(ch1)):
            if abs(ch1[i]) >= ch0[max_abs_ch1]*per_left:
                start_ch1 = i
                break

        for j in range(len(ch1)-1, 0, -1):
            if abs(ch1[j]) >= ch0[max_abs_ch0]*per_right:
                end_ch1 = j
                break
    else:
        for i in range(max_abs_ch0, 0, -1):
            if abs(ch0[i]) <= ch0[max_abs_ch0]*per_left:
                start_ch0 = i
                break

        for j in range(max_abs_ch0, len(ch0)):
            if abs(ch0[j]) <= ch0[max_abs_ch0]*per_right:
                end_ch0 = j
                break

        for i in range(max_abs_ch1, 0, -1):
            if abs(ch1[i]) <= ch0[max_abs_ch1]*per_left:
                start_ch1 = i
                break

        for j in range(max_abs_ch1, len(ch1)):
            if abs(ch1[j]) <= ch0[max_abs_ch1]*per_right:
                end_ch1 = j
                break

    ch0 = np.array(ch0[start_ch0: end_ch0])
    ch1 = np.array(ch1[start_ch1: end_ch1])
    word = []
    word.append(np.column_stack((ch0, ch1)))
    
    return word

#------------------------------------------------------------------------------
#Get value and index of start and end of word in sound signal

word = word_cut(filtered)

#TODO implementar el análisis de archivos .wav

#------------------------------------------------------------------------------
#Plot the filtered sound signals and the words
x = np.arange(0, t_size*dt, dt)
figure, axis = plt.subplots(2, 1)

axis[0].plot(x, filtered[t][:,0])
axis[0].set_title("Canal 1 (filtrado)")

axis[1].plot(x, filtered[t][:,1])
axis[1].set_title("Canal 2 (filtrado)")

plt.show()

#Plotting the words
x_word = np.arange(0, (len(word[0])+1)*dt-dt, dt)
figure, axis = plt.subplots(2, 1)

axis[0].plot(x_word, word[0][:,0])
axis[0].set_title("Palabra canal 1")

axis[1].plot(x_word, word[0][:,1])
axis[1].set_title("Palabra canal 2")

plt.show()

#------------------------------------------------------------------------------
#Play words
sd.play(filtered[t], fs)
sd.wait()

sd.play(word[0], fs)
sd.wait()