import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = '../clustering-benchmark/src/main/resources/datasets/artificial/'
databrut = arff.loadarff(open( path + "triangle1.arff" , 'r') )
datanp = [ [x[0] ,x[1]] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne

print ( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time ()
max_cluster = 10

silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
for k in range(2, max_cluster + 1):

    # Ajustez le modèle K-means
    model = cluster.KMeans(n_clusters =k , init ='k-means++')
    model.fit(datanp)
    labels = model.labels_

    # Calculez les métriques d'évaluation
    silhouette = silhouette_score(datanp, labels)
    davies_bouldin = davies_bouldin_score(datanp, labels)
    calinski_harabasz = calinski_harabasz_score(datanp, labels)

    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    calinski_harabasz_scores.append(calinski_harabasz)

# Tracez les résultats
plt.plot(range(2, max_cluster + 1), silhouette_scores, label='Silhouette Score')
plt.plot(range(2, max_cluster + 1), davies_bouldin_scores, label='Davies-Bouldin Score')
#plt.plot(range(2, max_cluster + 1), calinski_harabasz_scores, label='Calinski-Harabasz Score')

plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.legend()
plt.show()

# D'après la métrique d'evaluation de Calinski, on prend k=6 pour avoir le meilleur score
k = 4
model = cluster.KMeans(n_clusters =k , init ='k-means++')
model.fit(datanp)
tps2 = time.time ()
labels = model.labels_
iteration = model.n_iter_


print(f"nombre de clusters : {k} \n temps d'exécution : {round(1000*(tps2 - tps1))}")
plt.figure(figsize=(10, 5))

# Plot for initial data
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.scatter(f0, f1, s=8)
plt.title("Données initiales")

# Plot for data after KMeans clustering
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering KMeans")
plt.show()
