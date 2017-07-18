import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint

plt.rcParams['image.cmap']='jet'
df=pd.read_csv('../../InitialData/venture_capital_investment_dataset.csv')


def gimme_encodings():
	return { col_name: {"low":0, "medium":1, "high":2} for col_name in df[:-1] } 



#drop "?"
# that leaves the cells that contain ? as empty
df=df.replace('?', np.NaN)

#drop NA
#df=df.dropna()
#df.reset_index(drop=True)


#convert nominal variables into numerical
#df=pd.get_dummies(df)
#df.fillna(df.mean())

df=df.fillna(df.mode().iloc[0])
df.to_csv('encoded_capital_mode.csv', encoding='utf-8', index=False)


def plot_data(series, labels, filename):
	plt.figure()
	#last_score.hist()
	plt.xlabel(labels)
	series.plot.hist(alpha=0.9)
	hist_name='Plots/'+filename+'_hist.pdf'
	plt.savefig(hist_name, bbox_inches='tight')
	plt.show()

	plt.figure()
	color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
	series.plot.box(color=color, sym='r+')
	
	box_name='Plots/'+filename+'_box.pdf'
	plt.savefig(box_name, bbox_inches='tight')


def get_stats_graphs(series, labels, filename):

	##FB Statistics##
	print filename+" STATISTICS......."
	print "Average= ", series.mean()
	print "SD= ", series.std()
	print "MAD= ", abs(series-series.median()).median()
	plot_data(series, labels, filename)

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


	#plt.show()
	path= 'Plots/'+str(plot_name)+'_bar.pdf'
	plt.savefig(path, bbox_inches='tight')

def freq_tables(col_name):
	return pd.crosstab(df[col_name], columns="count").to_latex() 

def create_bars_tables():
	list_cols=list(df.columns.values)
	with open("table_file.txt","w'") as f:
		for name in list_cols:
			bar_chart(list(df[name].value_counts()), name)
			#print(df[name].value_counts())

			tb=freq_tables(name)
			f.write(tb)
			f.write('\n')

def extra_credit():
	encoded_df=pd.read_csv('../ProcessedData/encoded_capital_mode.csv')
	red_df=encoded_df[encoded_df['rating']=='red'].sample(frac=0.4).reset_index()
	print "#red=", len(red_df)

	green_df=encoded_df[encoded_df['rating']=='green'].sample(frac=0.7).reset_index()
	print "#green=",len(green_df)

	yello_df=encoded_df[encoded_df['rating']=='yellow'].sample(frac=1).reset_index()
	print "#yellow=", len(yello_df)

	#resample to shuffle data
	new_df=pd.concat([red_df, green_df, yello_df], axis=0).sample(frac=1).reset_index()
	print len(new_df)
	new_df.to_csv('balanced_classes.csv', encoding='utf-8', index=False)



if __name__ == '__main__':
	#plot_data(GRO, 'p', 'gro')
	#pprint.pprint(gimme_encodings())
	
	
	extra_credit()


	#df.replace(gimme_encodings(), inplace=True)
	#print df.head(5)


