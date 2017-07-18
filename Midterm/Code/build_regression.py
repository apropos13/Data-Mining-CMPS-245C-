from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

pd.options.display.float_format = '{:,.2g}'.format
df=pd.read_csv('../Data/filtered_no_missing_no_norm.csv')

#get variable to predict
y=df.current_satisfaction_score

X=df.drop(['current_satisfaction_score'], axis=1).astype('float64')
#print X.info()
X_train, X_test , y_train, y_test =train_test_split(X, y, test_size=0.5, random_state=True)

def least_squares():
	ridge1= Ridge(alpha=0, normalize=True)
	ridge1.fit(X_train, y_train)
	print 'Least Squares MSE on test', mean_squared_error(y_test, ridge1.predict(X_test))

	#test on all data
	diff= y- ridge1.predict(X)
	print "Max diff Least Squares=",diff.max()
	print "Min diff Least Squares=",diff.min()
	print "-------------------"

def ridge():
	alphas = 10**np.linspace(10,-2,100)*0.5

	ridge= Ridge(normalize=True)
	coefs=[]


	for a in alphas:
		ridge.set_params(alpha=a)
		ridge.fit(X, y)
		coefs.append(ridge.coef_)

	ax = plt.gca()
	ax.plot(alphas, coefs)
	ax.set_xscale('log')
	plt.axis('tight')
	plt.xlabel('alpha')
	plt.ylabel('weights')
	#plt.savefig('../Tex/wVSa.pdf',bbox_inches='tight')

	ridgecv= RidgeCV(alphas=alphas, scoring='mean_squared_error', normalize=True)
	ridgecv.fit(X,y)
	print "Ridge alpha=",ridgecv.alpha_

	ridge4= Ridge(alpha=ridgecv.alpha_, normalize=True)
	ridge4.fit(X_train, y_train)
	print 'Ridge MSE on test', mean_squared_error(y_test, ridge4.predict(X_test))
	#test on all data
	diff= y- ridge4.predict(X)
	print "Max diff Ridge=",diff.max()
	print "Min diff Ridge=",diff.min()
	print ridge4.coef_
	print "-------------------"

def elastic():
	elastic= ElasticNet(max_iter=10000, normalize=True)
	elasticCV= ElasticNetCV(alphas=None, cv=10, max_iter=100000, normalize=True)
	elasticCV.fit(X_train, y_train)
	elastic.set_params(alpha=elasticCV.alpha_)
	print "Elastc alpha=", elasticCV.alpha_
	elastic.fit(X_train, y_train)
	print "Elastic MSE on test= ", mean_squared_error(y_test, elastic.predict(X_test))

	#test on all data
	diff= y- elastic.predict(X)
	print "Max diff Elnet=",diff.max()
	print "Min diff Elnet=",diff.min()
	print "-------------------"



def lasso():
	lasso = Lasso(max_iter=10000, normalize=True)
	lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
	lassocv.fit(X_train, y_train)
	lasso.set_params(alpha=0.0000001)
	print "Lasso alpha=", lassocv.alpha_
	lasso.fit(X_train, y_train)
	print "Lasso MSE on test", mean_squared_error(y_test, lasso.predict(X_test))
	#test on all data
	diff= y_test- lasso.predict(X_test)
	print "Max diff Lasso=",diff.max()
	print "Min diff Lasso=",diff.min()
	print lasso.coef_
	print "-------------------"

if __name__ == '__main__':
	ridge()
	#least_squares()
	#lasso()
	#elastic()
	
