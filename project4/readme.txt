Project title:
[kernelizing k-means & three datasets]

Running:
With corresponding datasets(dataset1.csv, dataset2.csv, and dataset3.csv) in the same folder, simply run the scripts(kmeans_dataset1_K.py, kmeans_dataset2_K_pro.py, and kmeans_dataset3_K_pro.py) seperately. kmeans_dataset1_K.py corresponds to concentric circles, while kmeans_dataset2_K_pro.py, and kmeans_dataset3_K_pro.py correspond to two other additional datasets.
These three scripts use rbf kernel due to its hihgest precision compared to that of linear and quadratic kernel. To adjust the sigma value of rbf kernel, simply change the variable var. 
Due to the fact that we initialize the z value randomly, the precise result cannot be guaranteed for every epoch. Several rounds of trails have to be made before we get ideal clustering results.

Result:
After running the script, you will get a plot of dataset before clustering and a plot of resulting cluster.
