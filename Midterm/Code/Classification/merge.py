from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

df_sat=pd.read_csv('../../Data/filtered_clustered.csv')
df_turn=pd.read_csv('../../Data/employee_turnover_dataset.csv')

current_score=df_sat.current_satisfaction_score.astype(int)
last_score=df_sat.last_evaluation_satisfaction_score
n_projects=df_sat.number_projects
hours=df_sat.average_montly_hours
time=df_sat.time_spent_at_company
promo=df_sat.promotion_in_last_5_years.astype(int)
dep=df_sat.department.astype('category')
salary=df_sat.salary
ident=df_sat.id

lower=0.1
upper=0.9
current_score=current_score[(current_score>current_score.quantile(lower)) & (current_score<current_score.quantile(upper))]

last_score=last_score[(last_score>last_score.quantile(lower)) & (last_score<last_score.quantile(upper))]

n_projects=n_projects[(n_projects>n_projects.quantile(lower)) & (n_projects<n_projects.quantile(upper))]

hours=hours[(hours>hours.quantile(lower)) & (hours<hours.quantile(upper))]

time=time[(time>time.quantile(lower)) & (time<time.quantile(upper))]

promo[:]=0

salary=salary[(salary>salary.quantile(lower)) & (salary<salary.quantile(upper))]

print "Number of Categories=",dep.cat.codes.max()

dep_numerical= pd.get_dummies(df_sat['department'])

filtered_df= pd.concat([ident,last_score,n_projects,hours,time,salary,dep_numerical,current_score], axis=1)


df_stacked=pd.merge(filtered_df, df_turn, on='id')
df_stacked=df_stacked.dropna()

df_stacked.to_csv("../../Data/Classification/stacked.csv",sep=",", encoding='utf-8', index=False)


