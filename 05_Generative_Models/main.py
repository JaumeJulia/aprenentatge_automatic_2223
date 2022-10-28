import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


digits = datasets.load_digits()
plot_digits(digits.data[:100, :])

plt.show()
print(digits.data.shape)

#Aplicam el PCA

pca = PCA()
pca.fit(digits.data)

varCum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(varCum)
plt.show()

#aplanar la curva con una funcion

i = 0
for iter in varCum:
    i += 1
    if iter >= 0.95:
        n_components = varCum[:i]
        break

#Tornam a aplicar el PCA, ara amb les components necessaries

pca = PCA(n_components = len(n_components))
pca.fit(digits.data)
pcaDigits = pca.transform(digits.data)

varCum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(varCum)
plt.show()

#Usam ara el GaussianMixture model

bicData = []
minimo = 0
valorMinimo = 999999
for i in range(1, 11):
    gm = GaussianMixture(n_components=i)
    gm.fit(pcaDigits)
    bic = gm.bic(pcaDigits)
    bicData.append(bic)
    if valorMinimo > bic:
        valorMinimo = bic
        minimo = i

x = np.array(bicData)
plt.plot(x)
plt.show()

## Observant el gràfic, el valor mínim es 7 que es el valor de gaussianes que millor s'adapten

#Generam noves mostres

gm = GaussianMixture(n_components=minimo)
gm.fit(pcaDigits)
x, y = gm.sample(100)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

#Desfeim la transformació del PCA i mostram els resultats

x = pca.inverse_transform(x)
plot_digits(x)
plt.show()
