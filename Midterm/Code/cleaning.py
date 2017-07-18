from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../Data/filtered_clustered.csv')
current_score=df.current_satisfaction_score.astype(int)
last_score=df.last_evaluation_satisfaction_score
n_projects=df.number_projects
hours=df.average_montly_hours
time=df.time_spent_at_company
promo=df.promotion_in_last_5_years.astype(int)
dep=df.department.astype('category')
salary=df.salary



if __name__ == '__main__':

	lower=0.1
	upper=0.9
	current_score=current_score[(current_score>current_score.quantile(lower)) & (current_score<current_score.quantile(upper))]
	print "Curr Score:"
	print "Lower= ", current_score.quantile(lower)
	print "Upper= ",current_score.quantile(upper)


	last_score=last_score[(last_score>last_score.quantile(lower)) & (last_score<last_score.quantile(upper))]
	print "Last Score:"
	print "Lower= ", last_score.quantile(lower)
	print "Upper= ",last_score.quantile(upper)

	n_projects=n_projects[(n_projects>n_projects.quantile(lower)) & (n_projects<n_projects.quantile(upper))]
	print "proj:"
	print "Lower= ", n_projects.quantile(lower)
	print "Upper= ",n_projects.quantile(upper)

	hours=hours[(hours>hours.quantile(lower)) & (hours<hours.quantile(upper))]
	print "hours:"
	print "Lower= ", hours.quantile(lower)
	print "Upper= ",hours.quantile(upper)

	time=time[(time>time.quantile(lower)) & (time<time.quantile(upper))]
	print "time:"
	print "Lower= ", time.quantile(lower)
	print "Upper= ",time.quantile(upper)


	#promo=promo[(promo>promo.quantile(lower)) & (promo<promo.quantile(upper))]
	#print "promo:"
	#print "Lower= ", promo.quantile(lower)
	#print "Upper= ",promo.quantile(upper)
	promo[:]=0




	salary=salary[(salary>salary.quantile(lower)) & (salary<salary.quantile(upper))]
	print "salary:"
	print "Lower= ", salary.quantile(lower)
	print "Upper= ",salary.quantile(upper)

	for i in df.duplicated('id'):
		if i==True:
			print "FOund one"

	print df.isnull().values.any()

	print "Number of Categories=",dep.cat.codes.max()

	dep_numerical= pd.get_dummies(df['department'])

	filtered_df= pd.concat([last_score,n_projects,hours,time,salary,dep_numerical,current_score], axis=1)
	filtered_no_missing=filtered_df.dropna()
	filtered_no_missing.to_csv("../Data/filtered_no_missing_no_norm.csv", sep=',', encoding='utf-8', index=False)


	#do z score normalization
	current_score=(current_score-current_score.mean()) / current_score.std()
	print current_score.shape

	last_score=(last_score-last_score.mean()) / last_score.std()
	print last_score.shape

	n_projects=(n_projects-n_projects.mean()) / n_projects.std()
	hours=(hours-hours.mean()) / hours.std()
	time=(time-time.mean()) / time.std()
	salary=(salary-salary.mean()) / salary.std()

	filtered_df= pd.concat([last_score,n_projects,hours,time,salary,dep_numerical,current_score], axis=1)
	#filtered_df.to_csv("../Data/filtered_data.csv", sep=',')

	filtered_no_missing=filtered_df.dropna()
	filtered_no_missing.to_csv("../Data/filtered_no_missing.csv",sep=",", encoding='utf-8', index=False)



