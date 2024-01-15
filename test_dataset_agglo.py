
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
# set distance_threshold(0 ensures we compute the full tree )
#silhouette calvinski davies 
scores_name= ["silhouette", "calvinski", "davies"]
best_results = [[0],[0],[0]]
scores = [[], [], []]
dist_max = 100
step = 10
distances = np.arange(1, dist_max + 1, step)
for d in distances: 
    print(d/dist_max)
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering(distance_threshold = d ,linkage = 'complete' , n_clusters = None )
    model = model.fit(datanp )
    tps2 = time.time ()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    #check score

    silhouette = silhouette_score(datanp, labels)
    calinski = calinski_harabasz_score(datanp, labels)
    davies = davies_bouldin_score(datanp, labels)

    scores[0].append(silhouette)
    scores[1].append(calinski)
    scores[2].append(davies)


    if silhouette >= best_results[0][0] :
        best_results[0] = [silhouette, k, d, leaves, labels]
    if calinski >= best_results[1][0] :
        best_results[1] = [calinski, k, d, leaves, labels]
    if davies >= best_results[2][0] :
        best_results[2] = [davies, k, d, leaves, labels]

# Affichage clustering
plt.figure(figsize=(10,15))

for i in range(len(best_results)) : 
    [score, k, d, leaves, labels] = best_results[i]
    plt.subplot(3,1,1+i)
    plt.scatter(f0 , f1 , c = labels, s = 8 )
    plt.title(f"distance = {d} / nb clusters ={k} / nb feuilles = {leaves} score {scores_name[i]}")
    print("nb clusters =",k ,", nb feuilles =", leaves ,"runtime =", round (( tps2 - tps1 ) * 1000 , 2 ) ,"ms")


# Plotting each set of scores
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(distances, scores[0], marker='o', label='Silhouette')
plt.plot(distances, scores[2], marker='^', label='Davies-Bouldin')

plt.xlabel('Distance Threshold')
plt.ylabel('Score')
plt.title('Clustering Scores by Distance Threshold')
# Adding a legend
plt.legend()

plt.subplot(1,2,2)
plt.plot(distances, scores[1], marker='s', label='Calinski-Harabasz')
# Adding labels and title
plt.xlabel('Distance Threshold')
plt.ylabel('Score')
plt.title('Clustering Scores by Distance Threshold')
# Adding a legend
plt.legend()
print(scores[0][3])
plt.show()
'''
# set the number of clusters
k = 4
tps1 = time.time ()
model = cluster.AgglomerativeClustering(linkage = 'single' , n_clusters = 3)
model = model.fit(datanp )
tps2 = time.time ()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

plt.subplot(1,2,2)
plt.scatter(f0 , f1 , c = labels , s = 8 )
plt.title(f"Resultat du clustering avec {k} clusters ")
print("nb clusters =",k ,", nb feuilles =", leaves ,"runtime =", round (( tps2 - tps1 ) * 1000 , 2 ) ,"ms")'''
