import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
import scipy.cluster.hierarchy as shc
import os 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features(dimension 2 )
# Ex : [[ - 0.499261 , -0.0612356 ] ,
# [ - 1.51369 , 0.265446 ] ,
# [ - 1.60321 , 0.362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster.On retire cette information
path = '../clustering-benchmark/src/main/resources/datasets/artificial/'
databrut = arff.loadarff(open( path +"xclara.arff", 'r') )
datanp = [ [x[0] ,x[1]] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne

scores_name= ["silhouette", "calvinski", "davies"]
best_results = [[0],[0],[0]]
scores = [[], [], []]
max_cluster = 10

dist_max = 10
step = 1
distances = np.arange(2, dist_max + 1, step)

for k in range(2, max_cluster + 1):

    # Ajustez le modèle K-means
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering(linkage = 'average' , n_clusters = k)
    model = model.fit(datanp)
    tps2 = time.time ()
    labels = model.labels_
    #d=model.distances_
    leaves = model.n_leaves_

    # Calculez les métriques d'évaluation
    silhouette = silhouette_score(datanp, labels)
    davies_bouldin = davies_bouldin_score(datanp, labels)
    calinski_harabasz = calinski_harabasz_score(datanp, labels)

    scores[0].append(silhouette)
    scores[1].append(calinski_harabasz)
    scores[2].append(davies_bouldin)

    if silhouette >= best_results[0][0] :
        best_results[0] = [silhouette, k, leaves, labels]
    if calinski_harabasz >= best_results[1][0] :
        best_results[1] = [calinski_harabasz, k, leaves, labels]
    if davies_bouldin >= best_results[2][0] :
        best_results[2] = [davies_bouldin , k, leaves, labels]

# Affichage clustering
plt.figure(figsize=(10,15))

for i in range(len(best_results)) : 
    [score, k, leaves, labels] = best_results[i]
    plt.subplot(3,1,1+i)
    plt.scatter(f0 , f1 , c = labels, s = 8 )
    plt.title(f"nb clusters ={k} / nb feuilles = {leaves} score {scores_name[i]}")
    print("nb clusters =",k ,", nb feuilles =", leaves ,"runtime =", round (( tps2 - tps1 ) * 1000 , 2 ) ,"ms")

# Plotting each set of scores
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(distances, scores[0], marker='o', label='Silhouette')
plt.plot(distances, scores[2], marker='^', label='Davies-Bouldin')

plt.xlabel('Distance Threshold')
plt.ylabel('Score')
plt.title('Clustering Scores by K-Means')
# Adding a legend
plt.legend()

plt.subplot(1,2,2)
plt.plot(distances, scores[1], marker='s', label='Calinski-Harabasz')
# Adding labels and title
plt.xlabel('Distance Threshold')
plt.ylabel('Score')
plt.title('Clustering Scores by K-Means')
# Adding a legend
plt.legend()
print(scores[0][3])
plt.show()

