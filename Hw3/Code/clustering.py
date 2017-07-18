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
df= pd.read_csv("../Data/survey_dataset.csv")

#drop NA
df=df.dropna()
df.reset_index(drop=True)

#collect sections
music=df.iloc[:, 0:19]
movies=df.iloc[:, 19:31]
hobbies=df.iloc[:, 31:63]
phobias=df.iloc[:, 63:73]
health=df.iloc[:, 73:76]
personality=df.iloc[:, 76:133]
spending=df.iloc[:, 133:140]
demographics=df.iloc[:, 140:150]

#focus on one category 
selected_category=movies

#combine selected and demographics
dataset=pd.concat([selected_category, demographics], axis=1)
#encode categorical as numerical 
dataset=pd.get_dummies(dataset)

###################################

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x*1e-3)

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated to the last 40 steps)')
        #plt.xlabel('sample index or (cluster size)')
        plt.xlabel('cluster size', fontsize=16)
        plt.ylabel('distance', fontsize=16)
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def num_clusters(labels):
	# Number of clusters in labels, ignoring noise if present.
	return len(set(labels)) - (1 if -1 in labels else 0)

def eda():

	plt.figure(figsize=(20,10))
	plt.subplot(3,3,1)
	plt.scatter(dataset['Age'], dataset['Height'])
	plt.subplot(3,3,2)
	plt.hist(dataset['Age'])
	plt.subplot(3,3,3)
	sns.boxplot(dataset['Age'])	
	plt.show()

def reduce_dim(plot):
	#create TSNE representation
	tsne=TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
	dataset_tsne=tsne.fit_transform(dataset)
	dataset_tsne=dataset_tsne
	tsne_x=dataset_tsne[: ,0]
	tsne_y=dataset_tsne[: ,1]
	if plot:
		scatter= plt.scatter(tsne_x, tsne_y, alpha=0.75, s=100)
		plt.show()

	return tsne_x,tsne_y


def cluster(k, epsilon, distance_metric, tsneX, tsneY):
	#initialize models
	
	kmeans = KMeans(n_clusters=k, random_state=0)
	dbscan = DBSCAN(eps=epsilon)
	hierarchical = AgglomerativeClustering(n_clusters=k, affinity=distance_metric, linkage='ward')
	gaussian_mm= mixture.GaussianMixture(n_components=k)

	#fit data
	kmeans.fit(dataset)
	dbscan.fit(dataset)
	hierarchical.fit(dataset)
	gaussian_mm.fit(dataset)

	#get labels
	kmeans_labels=kmeans.labels_
	dbscan_labels=dbscan.labels_
	hierarchical_labels=hierarchical.labels_
	gaussian_labels=gaussian_mm.predict(dataset)

	#compute silhouette score: kmeans, hierarchical, gaussian
	kmeans_sil=metrics.silhouette_score(dataset, kmeans_labels, metric=distance_metric)
	hierarchical_sil=metrics.silhouette_score(dataset, hierarchical_labels, metric=distance_metric)
	gaussian_sil=metrics.silhouette_score(dataset, gaussian_labels, metric=distance_metric)

	#compute silhouette score: DBSCAN, need more than one cluster to get silhouette
	noise_index= np.argwhere(dbscan_labels==-1)
	if(np.unique(np.delete(dbscan_labels, noise_index)).size > 1):
		dbscan_sil=metrics.silhouette_score(dataset, dbscan_labels, metric=distance_metric)
	else:
		dbscan_sil=None

	#plot 
	plt.figure(figsize=(20,10))
	plt.subplot(2,2,1)
	plt.scatter(tsneX, tsneY, c=kmeans_labels)
	plt.title('KMeans \n Silhouette Score= %.4f' %(kmeans_sil))

	plt.subplot(2,2,2)
	plt.scatter(tsneX, tsneY, c=dbscan_labels)
	plt.title('DBScan \n Silhouette Score= %.4f' %(dbscan_sil))

	plt.subplot(2,2,3)
	plt.scatter(tsneX, tsneY, c=hierarchical_labels)
	plt.title('Hierarchical \n Silhouette Score= %.4f' %(hierarchical_sil))

	plt.subplot(2,2,4)
	plt.scatter(tsneX, tsneY, c=gaussian_labels)
	plt.title('Gaussian \n Silhouette Score= %.4f' %(gaussian_sil))


	plt.show()

def hierarchical(num_clusters, tsneX, tsneY, distance_metric, plot_dendro=False):
	print("####### AGGLO CLUSTERING #######")
	clusters=[c+2 for c in range(num_clusters)]
	silhouette_average=np.array([])
	silhouette_ward=np.array([])
	silhouette_complete=np.array([])

	print("Number of data points used= "), len(dataset)

	if plot_dendro:
		Z=linkage(dataset, 'ward')
		max_d=150 #set cutoff

		# calculate full dendrogram
		plt.figure(figsize=(25, 10))
		plt.title('Hierarchical Clustering Dendrogram')
		plt.xlabel('sample index')
		plt.ylabel('distance')
		fancy_dendrogram(
	    Z,
	    truncate_mode='lastp',
	    p=40,
	    leaf_rotation=90.,
	    leaf_font_size=12.,
	    show_contracted=True,
	    annotate_above=10,  # useful in small plots so annotations don't overlap
	    #max_d=max_d,  # plot a horizontal cut-off line
		)

		plt.savefig("dendro.pdf", bbox_inches='tight')
		plt.show()

	labels_average=[]
	labels_complete=[]
	labels_ward=[]	

	##keep the best clustering to get points##
	for n_clusters in clusters:
		for index, linkage_type in enumerate(('average', 'complete', 'ward')):
			agglom_labels = AgglomerativeClustering(linkage=linkage_type,
				n_clusters=n_clusters).fit_predict(dataset)

			silhouette_avg = metrics.silhouette_score(dataset, agglom_labels, metric=distance_metric)
			if index==0:
				silhouette_average=np.append(silhouette_average, silhouette_avg)
				labels_average.append(agglom_labels)
			elif index==1:
				silhouette_complete=np.append(silhouette_complete, silhouette_avg)
				labels_complete.append(agglom_labels)
			elif index==2:
				silhouette_ward=np.append(silhouette_ward, silhouette_avg)
				labels_ward.append(agglom_labels)

	plot_locations=[221, 222, 223, 224]

	n_best=np.argsort(silhouette_average)[::-1][:4] #get 4 best for average
	plt.figure(figsize=(20,10))
	plt.suptitle('Hierarchical with Average Linkage and Euclidean Distance', fontsize=16)
	for item,loc in zip(n_best, plot_locations):
		print("n_clusters :="+  str(clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_average[item]))
		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_average[item])
		plt.title('Clusters=%d \n Silhouette Score= %.4f' %(clusters[item], silhouette_average[item]))
	#plt.savefig("average_link.pdf")
	plt.show()

	#for complete also get the points in each cluster


	n_best=np.argsort(silhouette_complete)[::-1][:4] #get 4 best for complete
	plt.figure(figsize=(20,10))
	plt.suptitle('Hierarchical with Complete Linkage and Euclidean Distance', fontsize=16)
	for item,loc in zip(n_best, plot_locations):
		print("n_clusters :="+  str(clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_complete[item]))
		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_complete[item])
		plt.title('Clusters=%d \n Silhouette Score= %.4f' %(clusters[item], silhouette_complete[item]))
	plt.savefig("complete_link.pdf")
	plt.show()


	n_best=np.argsort(silhouette_ward)[::-1][:4] #get 4 best, ward
	plt.figure(figsize=(20,10))
	plt.suptitle('Hierarchical with Single Linkage and Euclidean Distance', fontsize=16)
	for item,loc in zip(n_best, plot_locations):
		print("n_clusters :="+  str(clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_ward[item]))
		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_ward[item])
		plt.title('Clusters=%d \n Silhouette Score= %.4f' %(clusters[item], silhouette_ward[item]))
	#plt.savefig("single_link.pdf")
	plt.show()

def kmeans(num_clusters, tsneX, tsneY, distance_metric, plot_sse=False):
	print("####### K-MEANS #######")
	clusters=[c+2 for c in range(num_clusters)]
	sse=[]
	centroids=[]
	silhouette_list=np.array([])
	labels_all=[]
	for c in clusters:
		print('computing cluster c:= '+str(c))
		kmeans=KMeans(n_clusters=c, random_state=0).fit(dataset)
		print('\t computing silhouette...')
		labels_list=kmeans.labels_
		labels_all.append(labels_list)
		silhouette_avg = metrics.silhouette_score(dataset, labels_list, metric=distance_metric)
		silhouette_list=np.append(silhouette_list, silhouette_avg)

		#print(kmeans.cluster_centers_)
		sse.append(kmeans.inertia_)

	n_best=np.argsort(silhouette_list)[::-1][:4] #get 4 best
	plot_locations=[221, 222, 223, 224]
	plt.figure(figsize=(20,10))
	plt.suptitle('KMeans', fontsize=16)
	for item,loc in zip(n_best, plot_locations):
		print("n_clusters :="+  str(clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_list[item]))
		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_all[item])
		plt.title('Clusters=%d \n Silhouette Score= %.4f' %(clusters[item], silhouette_list[item]))
	#plt.savefig("best_kmeans.pdf")
	plt.show()



	if plot_sse:
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.plot(clusters, sse, 'r-', 
			marker='o', 
			markersize=8, 
			markerfacecolor='red',
			linewidth=3, 
			label="K-Means Clustering",
			mec='black')

		ax1.grid(linestyle='-')
		plt.ylabel("SSE", fontsize=16)
		plt.xlabel("Number of clusters", fontsize=16)
		plt.xlim([min(clusters)-0.2, max(clusters)+.2])
		legend = ax1.legend(loc='upper right', shadow=True)
		ax1.yaxis.set_label_coords(-.10, 0.5)
		formatter = FuncFormatter(thousands)
		ax1.yaxis.set_major_formatter(formatter)
		#plt.savefig("kmeans.pdf", bbox_inches='tight')


		plt.show()

def dbscan(epsilon_list,tsneX, tsneY, distance_metric):
	print("####### DBSCAN CLUSTERING #######")
	print("Number of data points used= "), len(dataset)
	total_clusters=[]
	labels_all=[]
	#epsilon_list=np.arange(0.2,0.8,0.1)
	silhouette_list=np.array([])

	
	for e in epsilon_list:
		print("calculating dbscan for e:= "+str(e))
		labels_list= DBSCAN(eps=e, min_samples=5).fit_predict(dataset)
		total_clusters.append(num_clusters(labels_list))
		labels_all.append(labels_list)

		noise_index= np.argwhere(labels_list==-1)
		if(np.unique(np.delete(labels_list, noise_index)).size > 1):
			silhouette_avg=metrics.silhouette_score(dataset, labels_list, metric=distance_metric)
		else:
			silhouette_avg=None
		silhouette_list=np.append(silhouette_list, silhouette_avg)

	n_best=np.argsort(silhouette_list)[::-1][:4] #get 4 best
	plot_locations=[221, 222, 223, 224]
	plt.figure(figsize=(20,10))
	plt.suptitle('DBScan', fontsize=16)
	for item,loc,e in zip(n_best, plot_locations, epsilon_list):
		print("For e :="+str(epsilon_list[item]))
		print("\t n_clusters ="+  str(total_clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_list[item]))

		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_all[item])
		plt.title('Epsilon= %d,  Clusters=%d \n Silhouette Score= %.4f' %(e,total_clusters[item], silhouette_list[item]))
	#plt.savefig("best_dbscan_minpoints_5.pdf")
	plt.show()

def gaussian(num_clusters, tsneX, tsneY, distance_metric):
	print("####### Gaussian #######")
	clusters=[c+2 for c in range(num_clusters)]
	sse=[]
	centroids=[]
	silhouette_list=np.array([])
	labels_all=[]
	for c in clusters:
		print('computing cluster c:= '+str(c))
		gaussian_mm= mixture.GaussianMixture(n_components=num_clusters)
		gaussian_mm.fit(dataset)
		labels_list=gaussian_mm.predict(dataset)
		print('\t computing silhouette...')
		labels_all.append(labels_list)
		silhouette_avg = metrics.silhouette_score(dataset, labels_list, metric=distance_metric)
		silhouette_list=np.append(silhouette_list, silhouette_avg)


	n_best=np.argsort(silhouette_list)[::-1][:4] #get 4 best
	plot_locations=[221, 222, 223, 224]
	plt.figure(figsize=(20,10))
	plt.suptitle('Gaussian Mixture', fontsize=16)
	for item,loc in zip(n_best, plot_locations):
		print("n_clusters :="+  str(clusters[item]))
		print("\t average silhouette_score is ="+str(silhouette_list[item]))
		plt.subplot(loc)
		plt.scatter(tsneX, tsneY, c=labels_all[item])
		plt.title('Clusters=%d \n Silhouette Score= %.4f' %(clusters[item], silhouette_list[item]))
	#plt.savefig("best_gaussian.pdf")
	plt.show()

def get_cluster_points(estimator):
	#return dict of indices
	return {i: dataset.iloc[ np.where(estimator.labels_ == i)[0] ]  for i in range(estimator.n_clusters)}

if __name__ == '__main__':
	#eda()
	t_x, t_y= reduce_dim(False)
	#k=2
	#e=5
	metric='chebyshev'
	#cluster(k, e, 'euclidean', t_x, t_y)

#	kmeans(15, t_x, t_y, metric)
#	hierarchical(15, t_x, t_y, metric)
	epsilon_list=np.arange(2,30,1)
	dbscan(epsilon_list, t_x, t_y, metric)
#	gaussian(15, t_x, t_y, metric)


	
	#best hierarchical:

	#best_complete=AgglomerativeClustering(linkage='complete', n_clusters=2).fit(dataset)
	#pprint.pprint(get_cluster_points(best_complete)[1])

	#get_cluster_points(best_complete)[1].to_csv('cluster_1_pts.csv')
	#get_cluster_points(best_complete)[0].to_csv('cluster_2_pts.csv')