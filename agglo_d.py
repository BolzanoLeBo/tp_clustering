
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
import scipy.cluster.hierarchy as shc
import os 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import sys 
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
path = '../dataset-rapport/X1.txt'
# Open the file in read mode
databrut = np.loadtxt(path)
datanp = [ [x[0] ,x[1]] for x in databrut]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne


if len(sys.argv) > 1 : 
    linkage = sys.argv[1]
else : 
    linkage = 'average'


# Donnees dans datanp
'''print("Dendrogramme single donnees initiales")
linked_mat = shc.linkage(datanp , 'single')
plt.figure(figsize =(12 , 12 ) )
shc.dendrogram (linked_mat,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt.plot()
'''

scores_name= ["silhouette", "calvinski", "davies"]
best_results = [[0],[0],[9999]]
scores = [[], [], []]
d_min = 5000
d_max = 500000
step = 10000
distances = np.arange(d_min, d_max, step)
list_t = []
for d in distances: 
    print(100*(d-d_min)/(d_max-d_min),"%")
    
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering(distance_threshold = d ,linkage = linkage , n_clusters = None)
    labels = model.fit_predict(datanp)
    tps2 = time.time ()
    
    list_t.append(round (( tps2 - tps1 ) * 1000 , 2 ))
    
    k = model.n_clusters_
    leaves = model.n_leaves_
    #check score
    if k > 1 : 
        silhouette = silhouette_score(datanp, labels)
        calinski = calinski_harabasz_score(datanp, labels)
        davies = davies_bouldin_score(datanp, labels)

        scores[0].append(silhouette)
        scores[1].append(calinski)
        scores[2].append(davies)
    else : 

        silhouette = 0
        calinski = 0
        davies = 0

        scores[0].append(0)
        scores[1].append(0)
        scores[2].append(0)


    if silhouette >= best_results[0][0] :
        best_results[0] = [silhouette, k, d, leaves, labels]
    if calinski >= best_results[1][0] :
        best_results[1] = [calinski, k, d, leaves, labels]
    if davies <= best_results[2][0] :
        best_results[2] = [davies, k, d, leaves, labels]

# Affichage clustering

plt.figure(figsize=(18, 10))
plt.suptitle(f"Aglomerative Clustering Results by choosing dist for Linkage: {linkage}")

for i, (score, k, d, leaves, labels) in enumerate(best_results):
    plt.subplot(2, 3, i+1)  
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"nb clusters = {k} / dist = {d} / nb feuilles = {leaves} score {scores_name[i]}")

# Second row for the clustering scores and runtimes
# Silhouette and Davies-Bouldin scores
plt.subplot(2, 3, 4) 
plt.plot(distances, scores[0], marker='o', label='Silhouette')
plt.plot(distances, scores[2], marker='^', label='Davies-Bouldin')
plt.xlabel('distance')
plt.ylabel('Score')
plt.title('Clustering Scores')
plt.legend()

# Calinski-Harabasz score
plt.subplot(2, 3, 5) 
plt.plot(distances, scores[1], marker='s', label='Calinski-Harabasz')
plt.xlabel('distance')
plt.ylabel('Score')
plt.title('Clustering Scores')
plt.legend()

# Runtimes
plt.subplot(2, 3, 6) 
plt.plot(distances, list_t, marker='o')
plt.xlabel('distance')
plt.ylabel('Runtime (ms)')
plt.title('Runtimes')

plt.tight_layout()
plt.show()
