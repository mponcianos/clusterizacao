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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#dataset = pd.read_excel('dataset_2000.xls')
#dataset = pd.read_excel('dataset_38mil_sem_faz.xls')
dataset = pd.read_excel('dataset_15mil_cpb.xls')

X = dataset.iloc[:,3:8].values

# método do cotovelo => gráfico com curva para achar o K-ótimo
cotovelos = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=10)
    kmeans.fit(X)
    cotovelos.append(kmeans.inertia_)
    print(kmeans.inertia_)



plt.plot(range(1,11),cotovelos,c='b',marker="*")
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Cotovelos')
plt.show()

############################################################
# K-MEANS
# 
############################################################
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(X)

plt.scatter(X[:,0],X[:,1], marker='*', s=200, c=clusters)


# sumário de dispersão
pd.pivot_table(dataset, index=["competencia"], columns=clusters, values="peso-cor", aggfunc=np.count_nonzero)


############################################################
# AgglomerativeClustering
# 
############################################################
agg = AgglomerativeClustering(n_clusters=3,affinity="euclidean").fit_predict(X)
plt.scatter(X[:,0],X[:,1], s=200, c=agg)

# sumário de dispersão
pd.pivot_table(dataset, index=["competencia"], columns=agg, values="peso-cor", aggfunc=np.count_nonzero)

############################################################
# DBSCAN
# 
############################################################
dbscan = DBSCAN(eps=0.5, min_samples=150, metric='manhattan').fit_predict(X)
dbscan = DBSCAN(eps=0.5, min_samples=120, metric='euclidean').fit_predict(X)
plt.scatter(X[:,0],X[:,1], s=200, c=dbscan)
print (dbscan)

