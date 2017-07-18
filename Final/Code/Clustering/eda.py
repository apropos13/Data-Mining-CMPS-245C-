import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint
import itertools

plt.rcParams['image.cmap']='jet'
df=pd.read_csv('../../InitialData/crowdfunded_projects_dataset.csv')

#print the title of the duplicated movies
#print df[df.duplicated]['title']

#remove duplicates now since they are common to all columns
df=df.drop_duplicates()

title=df['title'].dropna()
descr=df['description'].dropna()
df=df.drop(['title', 'description'], axis=1)

#select numerical types 
num_df= df.select_dtypes(include=['int64'])

#select categorical types
categ_df= df.select_dtypes(include=['object'])
#print num_df.info()
#print categ_df.info()

##### NUMERICAL #######
def get_correlation_matrix(tofile, newdf=num_df):
	with open(tofile,"w'") as f:
		f.write(newdf.corr(method='pearson').to_latex())


def plot_data(series, labels, filename):
	plt.figure()
	#prj_goal.hist()
	plt.xlabel(labels, fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	series.plot.hist(alpha=0.9)
	hist_name='../ClusteringPlots/'+filename+'_hist.pdf'
	plt.savefig(hist_name, bbox_inches='tight')
	#plt.show()
	plt.figure()
	color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
	series.plot.box(color=color, sym='r+')
	
	min_v=series.min()
	max_v=series.max()
	print "maximum value= ", max_v
	plt.ylim([0,max_v+3])
	box_name='../ClusteringPlots/'+filename+'_box.pdf'
	plt.savefig(box_name, bbox_inches='tight')

def get_scatter_plots(series, filename):
	plt.figure()
	plt.scatter(num_df[series[0]], num_df[series[1]])
	plt.xlabel(series[0])
	plt.ylabel(series[1])
	plt.xlabel(series[0],fontsize=16)
	plt.ylabel(series[1],fontsize=16)
	scatter_name='../ClusteringPlots/'+filename+'_scatter.pdf'
	plt.savefig(scatter_name, bbox_inches='tight')


def get_stats_graphs(series, labels, filename):

	##FB Statistics##
	print filename+" STATISTICS......."
	print "Average= ", series.mean()
	print "SD= ", series.std()
	print "MAD= ", abs(series-series.median()).median()
	plot_data(series, labels, filename)

def eda_numerical():
	##### EDA NUMERICAL#####
	get_stats_graphs(num_df['amount_funded'], labels="Amount Funded", filename="amt")
	get_stats_graphs(num_df['project_goal'], labels="Project Goal", filename="prj")
	get_stats_graphs(num_df['number_backers'], labels="Number of Backers", filename="backers")
	get_correlation_matrix('corr_matrix.txt')


	for pair in itertools.combinations(list(num_df.columns.values), 2):
		get_scatter_plots(pair, filename=pair[0]+pair[1])


##### CATEGORICAL #####	
def bar_chart(bins,plot_name):
	

	ind=np.arange(len(bins)) #x location of data
	fig, ax = plt.subplots()
	my_width=0.15
	rects1 = ax.bar(ind, bins, width=my_width, color='b')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Count', fontsize=16)
	ax.set_title(plot_name, fontsize=16)
	ax.set_xticks(0.0011+ ind + my_width / 2)
	if plot_name=='rating':
		ax.set_xticklabels(('Red', 'Yellow', 'Green'))
	elif len(bins)==4:
		ax.set_xticklabels(('Missing', 'Low', 'Medium', 'High'))
	elif len(bins)==3:
		ax.set_xticklabels(('Low', 'Medium', 'High'))
	elif len(bins)==2:
		ax.set_xticklabels(('Low', 'Medium'))
	elif len(bins)==1:
		ax.set_xticklabels(('Low'))

	else:
		print("You re doing something wrong")
		print("size="+str(len(bins)))
		print(plot_name)

	ax.yaxis.set_label_coords(-.08, 0.5)

	for label in ax.xaxis.get_ticklabels():
	    # label is a Text instance
	    label.set_color('blue')
	    label.set_rotation(25)
	    label.set_fontsize(16)


	plt.show()
	path= 'Plots/'+str(plot_name)+'_bar.pdf'
	#plt.savefig(path, bbox_inches='tight')

def freq_tables(col_name, c_df=categ_df):
	largest_df= c_df[col_name].value_counts().nlargest(5)
	return pd.crosstab(largest_df, columns="count").to_latex() 

def create_bars_tables(c_df=categ_df):
	list_cols=list(c_df.columns.values)
	with open("table_file.txt","w" ) as f:
		for name in list_cols:
			#bar_chart(list(c_df[name].value_counts()), name)
			#print(c_df[name].value_counts())
			
			tb=freq_tables(name)
			print tb
			f.write(tb)
			f.write('\n')	
	
def filter_numerical(lower_bound, upper_bound):
	lower=lower_bound
	upper=upper_bound
	amount=num_df['amount_funded']
	prj_goal=num_df['project_goal']
	bakers=num_df['number_backers']


	amount=amount[(amount>amount.quantile(lower)) & (amount<amount.quantile(upper))]
	print "amount:"
	print "Lower= ", amount.quantile(lower)
	print "Upper= ",amount.quantile(upper)


	prj_goal=prj_goal[(prj_goal>prj_goal.quantile(lower)) & (prj_goal<prj_goal.quantile(upper))]
	print "prj:"
	print "Lower= ", prj_goal.quantile(lower)
	print "Upper= ",prj_goal.quantile(upper)


	bakers=bakers[(bakers>bakers.quantile(lower)) & (bakers<bakers.quantile(upper))]
	print "bakers:"
	print "Lower= ", bakers.quantile(lower)
	print "Upper= ",bakers.quantile(upper)

	

	#print df.isnull().values.any()

	#zscore normalize
	amount=(amount-amount.mean())/amount.std()
	prj_goal=(prj_goal-prj_goal.mean())/prj_goal.std()
	bakers=(bakers-bakers.mean())/bakers.std()

	filtered_df= pd.concat([amount, prj_goal, bakers], axis=1).dropna()

	print len(filtered_df)
	return filtered_df


	#print "Number of Categories=",dep.cat.codes.max()		

def top_categorical(top=10):
	location=categ_df['location']
	currency=categ_df['currency']
	category=categ_df['category']

	loc_largest=location.value_counts().nlargest(top).index.values
	curr_largest=currency.value_counts().nlargest(top).index.values
	cat_largest=category.value_counts().nlargest(top).index.values

	mask_loc= categ_df['location'].isin(loc_largest)
	categ_df['location'][~mask_loc]='other_loc'

	mask_curr=categ_df['currency'].isin(curr_largest)
	categ_df['currency'] [~mask_curr]='other_curr'

	mask_cat= categ_df['category'].isin(cat_largest)
	categ_df['category'][~mask_cat]='other_categ'


	dummies_loc=pd.get_dummies(categ_df['location'])
	dummies_curr= pd.get_dummies(categ_df['currency'])
	dummies_cat= pd.get_dummies(categ_df['category'])

	final_cat_df=pd.concat([dummies_loc, dummies_curr, dummies_cat],axis=1)
	return final_cat_df


if __name__ == '__main__':
	#print [categ_df['location'].value_counts()==1]
	#print categ_df['currency'].value_counts().nlargest(10)
	#print categ_df['category'].value_counts().nlargest(10)

	#print pd.concat( [categ_df['location'], categ_df['location'].value_counts()], axis=0)
	

	location= categ_df['location']
	currency= categ_df['currency']
	category= categ_df['category']

	filtered_num_df=filter_numerical(lower_bound=0.2, upper_bound=0.8)
	dummy_df=top_categorical(top=30)

	#strip non ascii
	descr= descr.apply(lambda x:''.join([i if 32 < ord(i) < 126 else " " for i in x]))
	title= title.apply(lambda x:''.join([i if 32 < ord(i) < 126 else " " for i in x]))
	location= location.apply(lambda x:''.join([i if 32 < ord(i) < 126 else " " for i in x]))
	currency= currency.apply(lambda x:''.join([i if 32 < ord(i) < 126 else " " for i in x]))
	category= category.apply(lambda x:''.join([i if 32 < ord(i) < 126 else " " for i in x]))

	#add the original values, since it will be easier to interpret the clusters later
	final_df= pd.concat([title, descr,location,currency,category,filtered_num_df, dummy_df], axis=1).dropna()
	final_df.replace('', np.nan, inplace=True)
	final_df.dropna(inplace=True)
	final_df.to_csv('filtered_clustering_30.csv', index=False, encoding='utf-8')




	
	



