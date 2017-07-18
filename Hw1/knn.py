
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as pl

all_data = pd.read_csv('clean.csv') # better be in the correct directory!

num_att=['movie_facebook_likes', 'imdb_score', 'budget']#numerical attrib.
target_att=['received_academy_award']


var=all_data.as_matrix(num_att).astype(float) #extract vars
target= all_data.as_matrix(target_att).astype(bool).ravel()

#1,5,10,20,50
k=500
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
knn.fit(var, target)
scores = cross_val_score(knn, var, target, cv=10)
#print scores

print("Euclidean Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='manhattan')
knn.fit(var, target)
scores = cross_val_score(knn, var, target, cv=10)
#print scores
print("Manhattan Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='chebyshev')
knn.fit(var, target)
scores = cross_val_score(knn, var, target, cv=10)
#print scores
print("Chebyshev Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski',p=5)
knn.fit(var, target)
scores = cross_val_score(knn, var, target, cv=10)
#print scores
print("Minkowski Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


'''
#extra stuffs
h = .02 # step size in the mesh

x_min, x_max = var[:,0].min() - .5, var[:,0].max() + .5
y_min, y_max = var[:,1].min() - .5, var[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4, 3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# Plot also the training points
pl.scatter(X[:,0], X[:,1],c=Y )
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

#pl.show()
'''
