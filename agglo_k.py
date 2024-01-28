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
path = '../dataset-rapport/x1.txt'
# Open the file in read mode
databrut = np.loadtxt(path)
datanp = [ [x[0] ,x[1]] for x in databrut  ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne

scores_name= ["silhouette", "calvinski", "davies"]
best_results = [[0],[0],[9999]]
scores = [[], [], []]

if len(sys.argv) > 1 : 
    linkage = sys.argv[1]
else : 
    linkage = 'average'

max_cluster = 50


n_clusters = np.arange(2, max_cluster + 1 )

#tableau contenant les temps d'exécution pour tous les k 
list_t = np.zeros(len(n_clusters))


for k in n_clusters:
    
    #print the progression
    k_min = n_clusters[0]
    k_max = n_clusters[-1]
    print(100*(k-k_min)/(k_max-k_min),"%")
    
    
    tps1 = time.time ()
    #création du model 
    model = cluster.AgglomerativeClustering(linkage = linkage , n_clusters = k)
    model = model.fit(datanp)
    tps2 = time.time ()
    
    list_t[k-2] = round (( tps2 - tps1 ) * 1000 , 2 )
    
    labels = model.labels_
    leaves = model.n_leaves_

    # Calculez les métriques d'évaluation
    silhouette = silhouette_score(datanp, labels)
    davies_bouldin = davies_bouldin_score(datanp, labels)
    calinski_harabasz = calinski_harabasz_score(datanp, labels)

    scores[0].append(silhouette)
    scores[1].append(calinski_harabasz)
    scores[2].append(davies_bouldin)

    
    #Trouver le meilleur score pour chacune des métrique d'évaluation 
    if silhouette >= best_results[0][0] :
        best_results[0] = [silhouette, k, leaves, labels]
    if calinski_harabasz >= best_results[1][0] :
        best_results[1] = [calinski_harabasz, k, leaves, labels]
    if davies_bouldin <= best_results[2][0] :
        best_results[2] = [davies_bouldin , k, leaves, labels]

# Affichage clustering
plt.figure(figsize=(18, 10))
plt.suptitle(f"Aglomerative Clustering Results by choosing k for Linkage: {linkage}")

for i, (score, k, leaves, labels) in enumerate(best_results):
    plt.subplot(2, 3, i+1)  
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"nb clusters = {k} / nb feuilles = {leaves} score {scores_name[i]}")

# Second row for the clustering scores and runtimes
# Silhouette and Davies-Bouldin scores
plt.subplot(2, 3, 4) 
plt.plot(n_clusters, scores[0], marker='o', label='Silhouette')
plt.plot(n_clusters, scores[2], marker='^', label='Davies-Bouldin')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('Clustering Scores')
plt.legend()

# Calinski-Harabasz score
plt.subplot(2, 3, 5) 
plt.plot(n_clusters, scores[1], marker='s', label='Calinski-Harabasz')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('Clustering Scores')
plt.legend()

# Runtimes
plt.subplot(2, 3, 6) 
plt.plot(n_clusters, list_t, marker='o')
plt.xlabel('k')
plt.ylabel('Runtime (ms)')
plt.title('Runtimes')

plt.tight_layout()
plt.show()


