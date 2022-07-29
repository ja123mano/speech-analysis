import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import noisereduce as nr
import sounddevice as sd
from scipy.ndimage import maximum_filter1d
from python_speech_features import mfcc, fbank, logfbank, ssc
from python_speech_features import delta
from sklearn.model_selection import KFold, cross_validate
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier



A1 = [ ["D:\\Decodificación del habla\\Datos\\S1\\S1_1.wav", "D:\\Decodificación del habla\\Datos\\S1\\S1_HP_cruces1.mrk", 1642014432149],
       ["D:\\Decodificación del habla\\Datos\\S1\\S1_2.wav", "D:\\Decodificación del habla\\Datos\\S1\\S1_HP_cruces2.mrk", 1642015279544],
       ["D:\\Decodificación del habla\\Datos\\S1\\S1_3.wav", "D:\\Decodificación del habla\\Datos\\S1\\S1_HP_cruces3.mrk", 1642018989749],
       ["D:\\Decodificación del habla\\Datos\\S1\\S1_4.wav", "D:\\Decodificación del habla\\Datos\\S1\\S1_HP_cruces4.mrk", 1642020887444]
      ] 

'''
["Archivo1.wav", "cruces.mrk", 1642014432149],
      ["Archivo2.wav", "cruces1.mrk", 1642015279544],
      ["Archivo3.wav", "cruces2.mrk", 1642018989749],
      ["Archivo4.wav", "cruces3.mrk", 1642020887444]
'''

'''
      ["S2_1.wav", "S2_HP_cruces1.mrk", 1642709615609],
      ["S2_2.wav", "S2_HP_cruces2.mrk", 1642711231738],
      ["S2_3.wav", "S2_HP_cruces3.mrk", 1642711835204],
      ["S2_4.wav", "S2_HP_cruces4.mrk", 1642712508208],
      ["S3_1.wav", "S3_HP_cruces1.mrk", 1644518789205],
      ["S3_2.wav", "S3_HP_cruces2.mrk", 1644520973370],
      ["S3_3.wav", "S3_HP_cruces3.mrk", 1644522282435],
      ["S3_4.wav", "S3_HP_cruces4.mrk", 1644523772595],
      ["S4_1.wav", "S4_HP_cruces1.mrk", 1644862510935],
      ["S4_2.wav", "S4_HP_cruces2.mrk", 1644863357729],
      ["S4_3.wav", "S4_HP_cruces3.mrk", 1644865356405],
      ["S4_4.wav", "S4_HP_cruces4.mrk", 1644866623555]
      ["S5_1.wav", "S5_HP_cruces1.mrk", 1645482826638],
      ["S5_2.wav", "S5_HP_cruces2.mrk", 1645483436673],
      ["S5_3.wav", "S5_HP_cruces3.mrk", 1645484657373],
      ["S5_4.wav", "S5_HP_cruces4.mrk", 1645485205833]
      
'''


#PROBAR CON LOS 11 SUJETOS
#AGREGAR TODAS LAS FEATURES
#AGREGAR MAS CLASIFICADORES(KNN)

i=0
matriz_x = []
matriz_y = []
while i in range(len(A1)):
    y = []
    dc = []
    to=A1[i][2]
    fs, Audiodata = wavfile.read(A1[i][0])
    with open(A1[i][1]) as archivo:
        for linea in archivo:
            if linea[0] == '1':
                tc = int(linea[2:len(linea)])  # rango
                dt = (tc - to) / 1000  # seg
                dcRes = int(dt * fs)
                dc.append(dcRes)

            if linea[0] == '2':
                y.append(2)
            if linea[0] == '3':
                y.append(3)
            if linea[0] == '4':
                y.append(4)
            if linea[0] == '5':
                y.append(5)
            if linea[0] == '6':
                y.append(6)



    nt = len(dc)
    #matriz = np.zeros([nt, 9 * fs])
    features = [] #mfcc
    features2 = [] #fbank
    features3 = [] #logfbank
    features4 = []#ssc
    duration = []
    audios = []

    for tr in range(nt):
        trial_audio = Audiodata[dc[tr]: (dc[tr] + 9 * fs), :]
        reduced_noise = nr.reduce_noise(trial_audio[:, 0], fs)
        r = reduced_noise
        r1 = maximum_filter1d(abs(r), size=1000)

        xm = np.argmax(r)
        mm = np.max(r)
        min_index = xm

        while True:
            min_index = min_index - 1
            if r1[min_index] < mm * 0.01:
                break

        max_index = xm
        while True:
            max_index = max_index + 1
            if r1[max_index] < mm * 0.01:
                break

        diferencia = max_index - min_index
        min_index2 = max_index + diferencia
        nueva_r1 = r[min_index:min_index2]
        duracion_palabra = (min_index2 - max_index) / fs

        mfcc_feat = mfcc(r, fs, nfft=2048)
        n_mfccs = np.array(mfcc_feat).flatten()
        features.append(n_mfccs)

        #NUEVAS FEATURES
        '''fbank_feat = fbank(r, fs, nfft=2048)
        n_fbank = np.array(fbank_feat).flatten()
        features2.append(n_fbank)'''

        logfbank_feat = logfbank(r, fs, nfft=2048)
        n_logfbank =  np.array(logfbank_feat).flatten()
        features3.append(n_logfbank)

        ssc_feat = ssc(r, fs, nfft=2048)
        n_ssc = np.array(ssc_feat).flatten()
        features4.append(n_ssc)

        duration.append(duracion_palabra)


        audios.append(trial_audio[min_index:min_index2, :])
        # print(features)
        # print("mfcc: ",mfcc_feat)

    x = np.array(features4)

    #print(y)
    i = i + 1

    matriz_x.append(x)
    matriz_y.append(y)

mat_x = np.concatenate(matriz_x, axis=0)
mat_y = np.concatenate(matriz_y, axis=0)
'''
print("duracion:", len(duration))
print("y:" ,len(y))

#clasificacion con matriz x y y
#SVM Lineal con cross validation
print("SVM Lineal")
classifier = svm.SVC(kernel = 'linear')
cv_results = cross_validate(classifier, mat_x,mat_y,cv = 5,scoring = ('accuracy', 'recall_micro'))
print('Acc: ', cv_results['test_accuracy'].sum()/5)
print('Recall: ', cv_results['test_recall_micro'].sum()/5)

#SVM de base radial
print("SVM de base radial")
clf_radial = svm.SVC(kernel = 'rbf')
scores = cross_validate(clf_radial, mat_x,mat_y,cv = 5,scoring = ('accuracy', 'recall_micro'))
print('Acc: ', cv_results['test_accuracy'].sum()/5)
print('Recall: ', cv_results['test_recall_micro'].sum()/5)

#knn
print("knn")
knn = KNeighborsClassifier(n_neighbors = 3)
cross_v = cross_validate(knn,mat_x,mat_y, cv= 10, scoring = ('accuracy', 'recall_micro'))
print('Acc: ', cross_v['test_accuracy'].mean())
print('Recall: ', cross_v['test_recall_micro'].mean())

#Decision tree
print("Decision tree")
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(mat_x, mat_y)
print(decision_tree_model.score(mat_x, mat_y))


#Perceptrón una capa
print("Perceptrón una capa")
one_perceptron_model = MLPClassifier(hidden_layer_sizes=(10), random_state=1, max_iter=10000)
one_perceptron_model.fit(mat_x, mat_y)
print(one_perceptron_model.score(mat_x, mat_y))

#Perceptrón multicapa
print("Perceptrón multicapa")
multiple_perceptron_model = MLPClassifier(hidden_layer_sizes=(10,10,10), random_state=1, max_iter=10000)
multiple_perceptron_model.fit(mat_x, mat_y)
print(multiple_perceptron_model.score(mat_x, mat_y))


n, bins, patches=plt.hist(duration)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()
'''


duration = np.array(duration)
y = np.array(y)



duration2 = duration[y == 2] #agua
print(duration2)

duration3 = duration[y == 3]#dormir

duration4 = duration[y == 4]#si

duration5 = duration[y == 5]#si

duration6 = duration[y == 6]#comida



n, bins, patches=plt.hist(duration2)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()

n, bins, patches=plt.hist(duration3)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()

n, bins, patches=plt.hist(duration4)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()

n, bins, patches=plt.hist(duration5)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()

n, bins, patches=plt.hist(duration6)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Duracion por palabra")
plt.show()




