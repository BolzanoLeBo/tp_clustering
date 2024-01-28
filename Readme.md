How to run : 

for kmean, you just have to launch "kmean.py" you can adjust the range of cluster inside the code by modifying the max_cluster variable 


For agglomerative there are two different codes: 

agglo_d.py choose the right number of cluster thanks to the distance treshold. 
You can modify the range of distance with the variables d_min, d_max and step 
You can also chaneg the linkage parameter when you launch the code from your bash 
ex : python agglo_d.py single will run agglo_d with the single linkage
by default the linkage is set to 'average' you can choose between : 
'average', 'single',  'complete', 'ward'

agglod_k.py make the number of cluster variate. You can also choose different linkages like in agglo_d


---------------------------------------------------------------------------------------------------

All the codes will plot :

graphics with the three scores (silhouette, calvinsk and davies)
a graphic with the runtime for each k or d 
A representation of the clustering output for the three best scores (silhouette, calvinski and davies)


---------------------------------------------------------------------------------------------------

You can choose which dataset to choose by modifying the path variable. 