import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('movie_dataset.csv')
fb_likes=df.movie_facebook_likes
imdb=df.imdb_score
budget=df.budget

##FB Statistics##
print "FB LIKES STATISTICS......."
print "Average= ", fb_likes.mean()
print "SD= ", fb_likes.std()
print "MAD= ", abs(fb_likes-fb_likes.median()).median()
print "Correlation with Num critic review= ", fb_likes.corr(df.num_critic_for_reviews)
print "Correlation with Actor_1_fb_likes= ", fb_likes.corr(df.actor_1_facebook_likes)
print "Correlation with Actor_2_facebook_likes= ", fb_likes.corr(df.actor_2_facebook_likes)
print "Correlation with Actor_3_fb_likes= ", fb_likes.corr(df.actor_3_facebook_likes)
print "Correlation with Gross= ", fb_likes.corr(df.gross)
print "Correlation with Num Voted Users= ", fb_likes.corr(df.num_voted_users)
print "Correlation with cast_total_facebook_likes= ", fb_likes.corr(df.cast_total_facebook_likes)
print "Correlation with facenumber_in_poster= ", fb_likes.corr(df.facenumber_in_poster)
print "Correlation with num_user_for_reviews= ", fb_likes.corr(df.num_user_for_reviews)
print "Correlation with budget= ", fb_likes.corr(df.budget)
print "Correlation with imdb_score= ", fb_likes.corr(df.imdb_score)
print "Correlation with aspect_ratio= ", fb_likes.corr(df.aspect_ratio)

plt.figure()
#imdb.hist()
plt.xlabel("FACEBOOK LIKES")
plt.xlim([0,200000])
fb_likes.plot.hist(alpha=0.9)
plt.show()

plt.figure()
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
fb_likes.plot.box(color=color, sym='r+')
plt.show()



## IMDB STATS ##
print "\n IMDB STATISTICS......."
print "Average= ", imdb.mean()
print "SD= ", imdb.std()
print "MAD= ", abs(imdb-imdb.median()).median()
print "Correlation with Num critic review= ", imdb.corr(df.num_critic_for_reviews)
print "Correlation with Actor_1_imdb= ", imdb.corr(df.actor_1_facebook_likes)
print "Correlation with Actor_2_facebook_likes= ", imdb.corr(df.actor_2_facebook_likes)
print "Correlation with Actor_3_imdb= ", imdb.corr(df.actor_3_facebook_likes)
print "Correlation with Gross= ", imdb.corr(df.gross)
print "Correlation with Num Voted Users= ", imdb.corr(df.num_voted_users)
print "Correlation with cast_total_facebook_likes= ", imdb.corr(df.cast_total_facebook_likes)
print "Correlation with facenumber_in_poster= ", imdb.corr(df.facenumber_in_poster)
print "Correlation with num_user_for_reviews= ", imdb.corr(df.num_user_for_reviews)
print "Correlation with budget= ", imdb.corr(df.budget)
print "Correlation with imdb= ", imdb.corr(df.movie_facebook_likes)
print "Correlation with aspect_ratio= ", imdb.corr(df.aspect_ratio)

plt.figure()
#imdb.hist()
plt.xlabel("IMDB SCORES")
imdb.plot.hist(alpha=0.9, bins=20)
plt.show()

plt.figure()
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
imdb.plot.box(color=color, sym='r+')
plt.show()



print " \n BUDGET STATISTICS......."
print "Average= ", budget.mean()
print "SD= ", budget.std()
print "MAD= ", abs(budget-budget.median()).median()
print "Correlation with Num critic review= ", budget.corr(df.num_critic_for_reviews)
print "Correlation with Actor_1_budget= ", budget.corr(df.actor_1_facebook_likes)
print "Correlation with Actor_2_facebook_likes= ", budget.corr(df.actor_2_facebook_likes)
print "Correlation with Actor_3_budget= ", budget.corr(df.actor_3_facebook_likes)
print "Correlation with Gross= ", budget.corr(df.gross)
print "Correlation with Num Voted Users= ", budget.corr(df.num_voted_users)
print "Correlation with cast_total_facebook_likes= ", budget.corr(df.cast_total_facebook_likes)
print "Correlation with facenumber_in_poster= ", budget.corr(df.facenumber_in_poster)
print "Correlation with num_user_for_reviews= ", budget.corr(df.num_user_for_reviews)
print "Correlation with budget= ", budget.corr(df.imdb_score)
print "Correlation with budget= ", budget.corr(df.movie_facebook_likes)
print "Correlation with aspect_ratio= ", budget.corr(df.aspect_ratio)

plt.figure()
#imdb.hist()
plt.xlabel("BUDGET")
plt.xlim([1000000,0.32e9])
budget.plot.hist(alpha=0.9,bins=200)
plt.show()

plt.figure()
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
budget.plot.box(color=color, sym='r+')
plt.show()
