import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

# TODO
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler

# TODO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entrenam una SVM linear (classe SVC)

# TODO

svm = SVC(C=1000, kernel="linear")
svm.fit(X_train_scaled, Y_train)

# Prediccio
# TODO

svm_predict = svm.predict(X_test)

# Metrica
# TODO

total = len(Y_test)
fails = np.nonzero(Y_test - svm_predict)
metric = (total - len(fails))/total
print(metric)
