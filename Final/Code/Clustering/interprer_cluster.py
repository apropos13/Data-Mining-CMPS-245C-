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
df_small= pd.read_csv("../../ProcessedData/cluster_2_pts.csv")
df_large= pd.read_csv("../../ProcessedData/cluster_1_pts.csv")

#drop NA
df_small=df_small.dropna()
df_small.reset_index(drop=True)
print("Size small", len(df_small))

df_large=df_large.dropna()
df_large.reset_index(drop=True)
print("Size large", len(df_large))

video_games=df_large[df_large['New York, NY']==1]
print float(len(video_games))


variances_small=df_small.var(axis=0)
variances_small.rename("small")
#print variances_small

variances_large=df_large.var(axis=0)
variances_large.rename("large")
#print variances_large

variances_diff= abs(variances_small-variances_large)
variances_diff.rename("diff")

variances=pd.concat([variances_small, variances_large, variances_diff], axis=1)
variances.columns=['cluster1', 'cluster2', 'diff']
variances=variances.sort_values('diff', ascending=False)
#print variances.head(15)


