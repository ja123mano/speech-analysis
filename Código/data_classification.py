#------------------------------------------------------------------------------------------------------------------
#   Speech data classification
#------------------------------------------------------------------------------------------------------------------

import pickle
import numpy as np
from scipy import signal

from python_speech_features import mfcc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

################################################################################
# Load audio records
fs = 44100

file_name = 'Datos\\07_22_2022_13_42_10.obj'
inputFile = open(file_name, 'rb')
data = pickle.load(inputFile)
n_trials = len(data)

# Filter signals
filt = signal.iirfilter(4, [10, 15000], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='ba')

filtered = []
for tr in data:
    ff1 = signal.filtfilt(filt[0], filt[1], tr[2][:,0], method='gust')
    ff2 = signal.filtfilt(filt[0], filt[1], tr[2][:,1], method='gust')
    filtered.append(np.column_stack((ff1, ff2)))


# Calculate MFCC features
features = []
for tr in filtered:
    mfcc_feat = mfcc(tr, fs, nfft = 2048)
    features.append(mfcc_feat.flatten())

# Build x and y arrays
x = np.array(features)
y = np.array([row[1] for row in data])

################################################################################
# Evaluate classification model

clf = SVC(kernel = 'linear')

n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle = True)

accuracy = 0
precision = np.zeros(6)
recall = np.zeros(6)
i = 0

for train_index, test_index in kf.split(x, y):

    print("********************************")

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)    

    accuracy_i = accuracy_score(y_test, y_pred)
    accuracy += accuracy_i
    print('Accuracy:', accuracy_i)

    precision_i = precision_score(y_test, y_pred, average = None)
    precision += precision_i
    print('Precision:', precision_i)

    recall_i = recall_score(y_test, y_pred, average = None)
    recall += recall_i
    print('Recall:', recall_i)
    
print("********************************")
print("Accuracy:", accuracy/n_folds)
print("Precision:", precision/n_folds)
print("Recall:", recall/n_folds)
print("\n\n-------------------------NEXT MODEL-------------------------\n\n")

################################################################################
# Evaluate classification model wtih feature selection

clf = SVC(kernel = 'linear')

n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle = True)

accuracy = 0
precision = np.zeros(6)
recall = np.zeros(6)

for train_index, test_index in kf.split(x, y):

    print("********************************")

    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]    

    # Feature selection
    ffs = SelectKBest(f_classif, k=2000)
    ffs.fit(x_train, y_train)
    x_train_new = ffs.transform(x_train)    
    x_test_new = ffs.transform(x_test)    

    # Training phase
    clf.fit(x_train_new, y_train)

    # Test phase    
    y_pred = clf.predict(x_test_new)
  
    accuracy_i = accuracy_score(y_test, y_pred)
    accuracy += accuracy_i
    print('Accuracy:', accuracy_i)

    precision_i = precision_score(y_test, y_pred, average = None)
    precision += precision_i
    print('Precision:', precision_i)

    recall_i = recall_score(y_test, y_pred, average = None)
    recall += recall_i
    print('Recall:', recall_i)    
    

print("********************************")
print("Accuracy:", accuracy/n_folds)
print("Precision:", precision/n_folds)
print("Recall:", recall/n_folds)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
