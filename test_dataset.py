import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
import scipy.cluster.hierarchy as shc
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

print("Appel KMeans pour une valeur fixee de k")
tps1 = time.time ()
k = 3
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



