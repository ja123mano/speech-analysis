import pickle
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import noisereduce as nr
import sounddevice as sd
from scipy.ndimage import maximum_filter1d

data_path = "D:\\Decodificación del habla\\Datos\\"

# audio_path = data_path + "S" + str(1) + "\\" + "S" + str(1) + "_" + str(1) + ".wav"
# marks_path = data_path + "S" + str(1) + "\\" + "S" + str(1) + "_HP_cruces" + str(1) + ".mrk"
# fs, data = wavfile.read(audio_path)

start_times = [[1642014432149, 1642015279544, 1642018989749, 1642020887444],
               [1642709615609, 1642711231738, 1642711835204, 1642712508208],
               [1644518789205, 1644520973370, 1644522282435, 1644523772595],
               [1644862510935, 1644863357729, 1644865356405, 1644866623555],
               [1645482826638, 1645483436673, 1645484657373, 1645485205833]]


#------------------------------------------------------------------------------
#Get the words identified by each code 
#and its start/finish time in ms from the .mrk file
def get_durations(marks_path, palabras={2: "Si", 3: "No", 4: "Agua", 5: "Comida", 6: "Dormir"}):
    durations = []

    with open(marks_path) as file:
        line_index = 0

        for line in file:
            marks = line[:line.rfind("\n")].split("\t")
            stage = int(marks[0])
            if stage != 1 and stage != 7:
                durations.append([stage, palabras.get(stage), int(marks[1]), 0])
            elif stage == 7:
                durations[line_index][-1] = int(marks[1])
                line_index += 1
    
    return durations

#------------------------------------------------------------------------------
#Get a list of lists of the words and its intensity data
#Slices the data retrieved from the audio file
#depending on the start/end times of the words in the durations array
def get_word_stages(audio_data, fs, durations, start_time):
    word_stages = []

    for i in range(len(durations)):
        start_cut = ((durations[i][2] - start_time)//1000)*fs
        end_cut = ((durations[i][3] - start_time)//1000)*fs
        word_stages.append(durations[i][:-2])
        word_stages[i].append(audio_data[start_cut:end_cut])

    return word_stages

#------------------------------------------------------------------------------
#Def for cutting words
#Cuts from the signal passed the left and right parts of it
#to_center = True, traverses the signal from the extremes to the center, stopping until the values found are >= max value of signal*percentage of each side
#to_center = False, traverses the signal from the center outwards, stopping until the values found are <= max value of the signal*percentage of each side
#If extra_ms_l/r is used, the cut word will get extra milliseconds before it starts or after it ends

#Returns the index where the cutting of the signal started and a slice of the original signal, cut from the moment a disruption in the intensity
#was found (depending on percentages given)

def word_cut(data, freq, palabra, per_left=0.1, per_right=0.1, to_center=True, extra_ms_l=0, extra_ms_r=0):
    if len(data[0]) > 3:
        ch0_o = data[palabra][:,0]
        ch1_o = data[palabra][:,1]
    else:
        ch0_o = data[palabra][2][0][:,0]
        ch1_o = data[palabra][2][0][:,1]
    
    ch0 = maximum_filter1d(abs(ch0_o), size=1000)
    ch1 = maximum_filter1d(abs(ch1_o), size=1000)

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
            if abs(ch1[j]) >= ch0[max_abs_ch1]*per_right:
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

    extra_index = 0
    if extra_ms_l > 0:
        extra_index = int((extra_ms_l/1000)*freq)
        start_ch0 -= extra_index
        start_ch1 -= extra_index

        if start_ch0 < 0:
            start_ch0 = 0
        if start_ch1 < 0:
            start_ch1 = 0
    
    if extra_ms_r > 0:
        extra_index = int((extra_ms_r/1000)*freq)
        end_ch0 += extra_index
        end_ch1 += extra_index

        if end_ch0 >= len(ch0_o):
            end_ch0 = len(ch0_o)-1
        if end_ch1 >= len(ch1_o):
            end_ch1 = len(ch1_o)-1

    start_both = min(start_ch0, start_ch1)
    end_both = max(end_ch0, end_ch1)
    ch0 = np.array(ch0_o[start_both: end_both])
    ch1 = np.array(ch1_o[start_both: end_both])
    word = []
    word.append(np.column_stack((ch0, ch1)))
    
    return start_both, word

#------------------------------------------------------------------------------
#Gives a list of lists which contains (each one) the cut words from all the wav files
#and the indexes of when the word started to be pronounced
def cut_all_words(data_dir):
    cut_all_words = []
    cutting_indexes = []

    for person in range(1, 6):
        for wav in range(1, 5):
            print(f"Persona {person}, Audio {wav}")
            audio_dir = data_dir + "S" + str(person) + "\\" + "S" + str(person) + "_" + str(wav) + ".wav"
            marks_dir = data_dir + "S" + str(person) + "\\" + "S" + str(person) + "_HP_cruces" + str(wav) + ".mrk"

            freq, data = wavfile.read(audio_dir)
            durations = get_durations(marks_dir)
            words = get_word_stages(data, freq, durations, start_times[person-1][wav-1])

            filtered_words = []
            for tr in range(len(words)):
                ff1 = nr.reduce_noise(words[tr][2][:,0], freq)
                ff2 = nr.reduce_noise(words[tr][2][:,1], freq)
                filtered_words.append(words[tr][:-1])
                filtered_words[tr].append([np.column_stack((ff1, ff2))])

            cut_words = []
            temp_list = [0]
            for palabra in range(len(filtered_words)):
                temp_list[0] = filtered_words[palabra][0]
                ind, cut_words_temp = word_cut(filtered_words, freq, palabra, per_right=0.01, to_center=False, extra_ms_r=50)
                cutting_indexes.append(ind)
                cut_words.append(temp_list.copy())
                cut_words[palabra].append(cut_words_temp)

            cut_all_words.append(cut_words)

    return cut_all_words, cutting_indexes

#------------------------------------------------------------------------------
#Gives a list with lists containing the durations of each word and the duration of the silence before the start of the word
def duration_all_words(words, cutting_indexes, freq):
    all_durations = []

    word2 = []
    word3 = []
    word4 = []
    word5 = []
    word6 = []

    silence2 = []
    silence3 = []
    silence4 = []
    silence5 = []
    silence6 = []

    duration = 0.0
    dt = 1/freq
    for palabra in range(len(words)):
        duration = (len(words[palabra][1][0]))*dt*1000
        duration_start_silence = cutting_indexes[palabra]*dt*1000

        if words[palabra][0] == 2:
            word2.append(duration)
            silence2.append(duration_start_silence)
        elif words[palabra][0] == 3:
            word3.append(duration)
            silence3.append(duration_start_silence)
        elif words[palabra][0] == 4:
            word4.append(duration)
            silence4.append(duration_start_silence)
        elif words[palabra][0] == 5:
            word5.append(duration)
            silence5.append(duration_start_silence)
        elif words[palabra][0] == 6:
            word6.append(duration)
            silence6.append(duration_start_silence)

    all_durations.append((word2, silence2))
    all_durations.append((word3, silence3))
    all_durations.append((word4, silence4))
    all_durations.append((word5, silence5))
    all_durations.append((word6, silence6))

    return all_durations

#------------------------------------------------------------------------------
#Get the durations and word data from the .wav file
# durations = get_durations(marks_path)
# word_stages = get_word_stages(data, fs, durations, start_times[0][0])

#------------------------------------------------------------------------------
#Plot words
# n = 3
# t_size = word_stages[n][2].shape[0]
# dt = 1/fs
# word_sliced_filtered = word_cut(filtered, fs, n, per_right=0.01, to_center=False, extra_ms_r=50)

# UNFILTERED CUT WORD DATA
#
# word_sliced_unfiltered = word_cut(word_stages, n, to_center=False)

# x = np.arange(0, t_size*dt, dt)
# figure, axis = plt.subplots(2, 1)
# axis[0].plot(x, word_stages[n][2][:,0])
# axis[0].set_title("Canal 1")
# axis[1].plot(x, word_stages[n][2][:,1])
# axis[1].set_title("Canal 2")
# plt.show()

# # PLOT OF THE UNFILTERED CUT WORD
# #
# # x_word = np.arange(0, (len(word_sliced_unfiltered[0])+1)*dt-dt, dt)
# # figure, axis = plt.subplots(2, 1)
# # axis[0].plot(x_word, word_sliced_unfiltered[0][:,0])
# # axis[0].set_title("Canal 1 - cortado")
# # axis[1].plot(x_word, word_sliced_unfiltered[0][:,1])
# # axis[1].set_title("Canal 2 - cortado")
# # plt.show()

# x_word_filtered = np.arange(0, (len(word_sliced_filtered[0])+1)*dt-dt, dt)
# figure, axis = plt.subplots(2, 1)
# axis[0].plot(x_word_filtered, word_sliced_filtered[0][:,0])
# axis[0].set_title("Canal 1 - cortado filtrado")
# axis[1].plot(x_word_filtered, word_sliced_filtered[0][:,1])
# axis[1].set_title("Canal 2 - cortado filtrado")
# plt.show()

#------------------------------------------------------------------------------
#Play words

# sd.play(word_stages[n][2], fs)
# sd.wait()

# # UNFILTERED CUT WORD SOUND
# #
# # sd.play(word_sliced_unfiltered[0], fs)
# # sd.wait()

# sd.play(word_sliced_filtered[0], fs)
# sd.wait()

#------------------------------------------------------------------------------
#Get and save the durations of the cut words from the wav files and the durations of the silences before them

# total_durations = duration_all_words(data_path)

# hist_file = open("D:\\Decodificación del habla\\hist_wordDuration2_data.obj", "wb")
# pickle.dump(total_durations, hist_file)
# hist_file.close()

#------------------------------------------------------------------------------

all_cut_words, indexes = cut_all_words(data_path)

hist_file = open("D:\\Decodificación del habla\\hist_cutWords_data.obj", "wb")
pickle.dump(all_cut_words, hist_file)
hist_file.close()

#------------------------------------------------------------------------------


"""
TODO Introduccion a scikit learning y clasificacion
TODO Clasificacion en machine learning | simplilearn    https://www.youtube.com/watch?v=xG-E--Ak5jg&ab_channel=Simplilearn
TODO Más tutoriales                                     https://www.youtube.com/watch?v=gJo0uNL-5Qw&ab_channel=codebasics
TODO Ver mfcc y entender sus características
"""