import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
wine = pd.read_csv("D:\\ExcelR Data\\Assignments\\PCA\\wine.csv")
wine.describe()
wine.head()

# Normalizing the numerical data 
wine_normal = scale(wine)

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
#Weight of the principle components
pca.components_[0]
#From above analysis see,First PC1 conatains 39% of data's and PC2 contains 17% of data's, PC3 contains 10% of data's etc...
# Calculating variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
plt.scatter(x,y,color=["red"])

#Clustering


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

newdf = pd.DataFrame(pca_values[:,0:3])

k=list(range(1,15))#here I'm defining my cluster range from 1 to 14

#Next I need to identify the total sum of square using TWSS[] 
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(newdf)
    WSS=[]#With in sum of squares
    for j in range(i):
         WSS.append(sum(cdist(newdf.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,newdf.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
#from the screeplot I can see number 12 is my Elbo point
#So I'm going to choose my cluster number as 12

model=KMeans(n_clusters=12) 
model.fit(newdf)

model.labels_     

md=pd.Series(model.labels_)
md.head()#first five label of cluster

md.tail()#last five label of cluster

#Hierarchical clustering
from scipy.cluster.hierarchy import linkage#here I'm importing the linkage function from hierarchy of cluster from scipy module
#for seeing dendrogram i'm going to hierarchy as sch
import scipy.cluster.hierarchy as sch
C = linkage(newdf,method="complete",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
        C,
        leaf_rotation=0.,
        leaf_font_size=8.,
)
plt.show()
#Before I calculated cluster number as 12 from the screeplot,Here also I'm going to use this same number of clusters as 10
#Impoting the agglomerative clustering for how many clusters that we need to see or cut
from sklearn.cluster import AgglomerativeClustering
C_linkage=AgglomerativeClustering(n_clusters=12,linkage="complete",affinity="euclidean").fit(newdf)
#labels of clusters
C_linkage.labels_

cluster_labels=pd.Series(C_linkage.labels_)#here transforming the series of labels to a new a dataset called "cluster_labels"

