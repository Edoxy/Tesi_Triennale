import pandas as pd
import numpy as np

from scipy.sparse.construct import random
from sklearn import datasets

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from IPython.display import display
from FisherDA import MultipleFisherDiscriminantAnalysis as MDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 

import os, ssl, sys


sys.stdout = open('/MDAvsPCA.txt', 'a')

lfw_people = datasets.fetch_lfw_people(min_faces_per_person=100, resize=0.5)

face_data = lfw_people['data']
face_images = lfw_people['images']
face_tnames = lfw_people['target_names']
face_targets = lfw_people['target']

# Creare gli X_trainval, y_trainval, X_test, y_test
# (RICORDA: il validation set viene creato "internamente" dalla classe MLPClassifier. 
# Gli deve essere solamente specificata la percentuale rispetto al training set)

random_state = np.random.randint(0, 100000)
print('random_seed = ', random_state)
test_p = 0.35
val_p = 0.25  # Percentuale di dati di X_trainval da usare come validation set


X_trainval, X_test, y_trainval, y_test, _, img_test = train_test_split(face_data, face_targets, face_images, test_size=test_p, random_state=random_state, shuffle=True)

display(pd.DataFrame({'X_trainval': X_trainval.shape, 'X_test': X_test.shape}, index=['N. sanmples', 'N.features']))

# Preparazione PCA

n_comp_fda = X_trainval.shape[0] - face_tnames.shape[0]

pca = PCA(n_components= n_comp_fda)
pca.fit(X_trainval)

X_trainval_old = X_trainval.copy()
X_trainval = pca.transform(X_trainval)

X_test_old = X_test.copy()
X_test = pca.transform(X_test)

mda = LDA(solver='eigen', n_components=4)
mda.fit(X_trainval, y_trainval)


# Trasformazione dati. Salvare i vecchi in "copie di backup"

X_trainval_mda = mda.transform(X_trainval)
X_test_mda = mda.transform(X_test)

#oversampling
ros = RandomOverSampler(random_state= random_state)
y_trainval_old = y_trainval.copy()
X_trainval, y_trainval = ros.fit_resample(X_trainval, y_trainval)
X_trainval_mda, y_trainval_mda = ros.fit_resample(X_trainval_mda, y_trainval_old)


# Inizializzazione iper-parametri MLP
hidden_layer_sizes = [10]
activation = 'relu'
#300
patience = 300
#1000
max_epochs = 5000
verbose = False
batch_sz = 4

# Inizializzazione MLP
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, batch_size=batch_sz, max_iter=max_epochs, early_stopping=True, n_iter_no_change=patience, random_state=random_state ,validation_fraction=val_p)

print(mlp.fit(X_trainval_mda, y_trainval_mda))

# Performance

y_pred_trainval = mlp.predict(X_trainval_mda)
y_pred = mlp.predict(X_test_mda)

acc_trainval = mlp.score(X_trainval_mda, y_trainval_mda)
prec_trainval = precision_score(y_trainval_mda, y_pred_trainval, average='weighted')
rec_trainval = recall_score(y_trainval_mda, y_pred_trainval, average='weighted')
f1_trainval = f1_score(y_trainval_mda, y_pred_trainval, average='weighted')

acc = mlp.score(X_test_mda, y_test)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

df_perf = pd.DataFrame({'Accuracy': [acc_trainval, acc], 
                        'Precision': [prec_trainval, prec], 
                        'Recall': [rec_trainval, rec],
                        'F1': [f1_trainval, f1]
                       },
                      index=['train. + val.', 'test'])

cmat = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
cmat_norm_true = confusion_matrix(y_test, y_pred, labels=mlp.classes_, normalize='true')
cmat_norm_pred = confusion_matrix(y_test, y_pred, labels=mlp.classes_, normalize='pred')

df_cmat = pd.DataFrame(cmat, columns=face_tnames, index=face_tnames)
df_cmat_norm_true = pd.DataFrame(cmat_norm_true, columns=face_tnames, index=face_tnames)
df_cmat_norm_pred = pd.DataFrame(cmat_norm_pred, columns=face_tnames, index=face_tnames)

display(df_perf)
display(df_cmat)
display(df_cmat_norm_true)
display(df_cmat_norm_pred)

print("\n")
sys.stdout.close()