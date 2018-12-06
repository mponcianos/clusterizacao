#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UFSC - Aprendizado de Máquina
Prof.: Gustavo Medeiros
CRIADO 05/12/2018
@author: Marcio Ponciano
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('dataset_2000.xls')

X = dataset.iloc[:,3:8].values

# método do cotovelo => gráfico com curva para achar o K-ótimo
from sklearn.cluster import KMeans

cotovelos = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=10)
    kmeans.fit(X)
    cotovelos.append(kmeans.inertia_)

plt.plot(range(1,11),cotovelos,c='b',marker="*")
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Cotovelos')
plt.show()

# Aplicar número de clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(X)

plt.scatter(X[:,0],X[:,1], marker='*', s=200, c=clusters)
plt.show()

# sumário de dispersão
pd.pivot_table(dataset, index=["competencia"], columns=clusters, values="peso-cor", aggfunc=np.count_nonzero)












