import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
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
path = '../dataset-rapport/zz1.txt'
# Open the file in read mode
databrut = np.loadtxt(path)
datanp = [ [x[0] ,x[1]] for x in databrut  ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
f0 = [x[0] for x in datanp]  # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne


# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ - 0.0612356 , 0.265446 , 0.362039 , ...]
#f0 = [x[0] for x in array_0]  # tous les elements de la premiere colonne
#f1 = [x[1] for x in array_1] # tous les elements de la deuxieme colonne

tps1 = time.time ()
max_cluster = 30

scores_name= ["silhouette", "calvinski", "davies"]
best_results = [[0],[0],[9999]]
scores = [[], [], []]
n_clusters = np.arange(2, max_cluster + 1 )

#tableau contenant les temps d'exécution pour tous les k 
list_t = np.zeros(len(n_clusters))


for k in n_clusters:
    #print the progression
    k_min = n_clusters[0]
    k_max = n_clusters[-1]
    print(100*(k-k_min)/(k_max-k_min),"%")

    # Ajustez le modèle K-means
    model = cluster.KMeans(n_clusters =k , init ='k-means++')
    tps1 = time.time ()
    model.fit(datanp)
    tps2 = time.time ()
    
    list_t[k-2] = round (( tps2 - tps1 ) * 1000 , 2 )
    labels = model.labels_


    # Calculez les métriques d'évaluation
    silhouette = silhouette_score(datanp, labels)
    davies_bouldin = davies_bouldin_score(datanp, labels)
    calinski_harabasz = calinski_harabasz_score(datanp, labels)

    scores[0].append(silhouette)
    scores[1].append(calinski_harabasz)
    scores[2].append(davies_bouldin)

    
    #Trouver le meilleur score pour chacune des métrique d'évaluation 
    if silhouette >= best_results[0][0] :
        best_results[0] = [silhouette, k, labels]
    if calinski_harabasz >= best_results[1][0] :
        best_results[1] = [calinski_harabasz, k, labels]
    if davies_bouldin <= best_results[2][0] :
        best_results[2] = [davies_bouldin , k, labels]

# Affichage clustering
plt.figure(figsize=(18, 10))
plt.suptitle(f"kmean results")

for i, (score, k, labels) in enumerate(best_results):
    plt.subplot(2, 3, i+1) 
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"nb clusters = {k} / score {scores_name[i]} : {score}")

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

