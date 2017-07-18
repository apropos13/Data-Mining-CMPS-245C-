import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import cluster
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn import metrics 
from matplotlib.ticker import FuncFormatter

import pprint

plt.rcParams['image.cmap']='jet'

#load dataset
df_small= pd.read_csv("../Data/cluster_1_pts.csv")
df_large= pd.read_csv("../Data/cluster_2_pts.csv")

#drop NA
df_small=df_small.dropna()
df_small.reset_index(drop=True)

df_large=df_large.dropna()
df_large.reset_index(drop=True)



variances_small=df_small.var(axis=0)
#print variances_small

variances_large=df_large.var(axis=0)
#print variances_large

variances=pd.concat([variances_small, variances_large], axis=1)
print variances


