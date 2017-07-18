import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('../Data/employee_satisfaction_dataset.csv')
current_score=df.current_satisfaction_score.astype(int)
last_score=df.last_evaluation_satisfaction_score
n_projects=df.number_projects
hours=df.average_montly_hours
time=df.time_spent_at_company
promo=df.promotion_in_last_5_years
dep=df.department
salary=df.salary

with open("table_file.txt","w'") as f:
	f.write(df.corr(method='pearson').to_latex())


def plot_data(series, labels, filename):
	plt.figure()
	#last_score.hist()
	plt.xlabel(labels)
	series.plot.hist(alpha=0.9)
	hist_name='Plots/'+filename+'_hist.pdf'
	plt.savefig(hist_name, bbox_inches='tight')
	#plt.show()
	plt.figure()
	color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
	series.plot.box(color=color, sym='r+')
	
	min_v=series.min()
	max_v=series.max()
	plt.ylim([0,max_v+3])
	box_name='Plots/'+filename+'_box.pdf'
	plt.savefig(box_name, bbox_inches='tight')


def get_stats_graphs(series, labels, filename):

	##FB Statistics##
	print filename+" STATISTICS......."
	print "Average= ", series.mean()
	print "SD= ", series.std()
	print "MAD= ", abs(series-series.median()).median()
	plot_data(series, labels, filename)


if __name__ == '__main__':
	#get_stats_graphs(current_score, "Current Satisfaction Score", "curr")
	#get_stats_graphs(last_score, "Last Evaluation Satisfaction Score", "last")
	#get_stats_graphs(n_projects, "Number of Projects", "proj")
	#get_stats_graphs(hours, "Monthly Hours", "hours")
	#get_stats_graphs(time, "Time Spent at Company", "time")
	#get_stats_graphs(promo, "Promotion in last 5 years", "promo")
	#get_stats_graphs(dep, "Departement", "promo")
	get_stats_graphs(salary, "Salary", "salary")


