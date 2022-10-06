import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.datasets import  make_classification
from Adaline import Adaline

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
standardData = scaler.transform(X)

#print(standardData)

# TODO: Entrenar usant l'algorisme de Batch gradient

batchPerceptron = Adaline()
y_prediction = batchPerceptron.fit(standardData, y)

# TODO: Mostrar els resultats

###  Mostram els resultats
plt.figure(1)
# Dibuixam el núvol de punts (el parametre c indica que colorejam segons la classe)
plt.scatter(X[:, 0], X[:, 1], c=y)

# Dibuixem la recta. Usam l'equació punt-pendent
m = -batchPerceptron.w_[1] / batchPerceptron.w_[2]
origen = (0, -batchPerceptron.w_[0] / batchPerceptron.w_[2])
plt.axline(xy1=origen, slope=m)

### Extra: Dibuixam el nombre d'errors en cada iteracio de l'algorisme
plt.figure(2)
plt.plot(batchPerceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error quadràtico mediano')
plt.show()
