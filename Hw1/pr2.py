import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df=pd.read_csv('movie_dataset.csv')
fb_likes=df.movie_facebook_likes
imdb=df.imdb_score
budget=df.budget


movie_dup=df.duplicated(["movie_title"])
print "Num of duplicate movie titles= ",len(movie_dup[movie_dup==True])


dup= df.duplicated()

print "Num of total duplicate rows= ",len(dup[dup==True])


df=df.drop_duplicates()
df.dropna(how='any',inplace=True)

c=0
for e in df['movie_facebook_likes']:
	if e==np.nan:
		c+=1

print "C= ", c

new_movie_dup=df.duplicated(["movie_title","plot_keywords"])
print "New Num of duplicate movie titles= ",len(new_movie_dup[new_movie_dup==True])

#after dropping duplicates filter quantiles

q1_fb= df['movie_facebook_likes'].quantile(0.05)
q2_fb= df['movie_facebook_likes'].quantile(0.95)
print "q1 =", q1_fb
print "q2 =" ,q2_fb

#filter fb
lower_fb= df[ (df['movie_facebook_likes']>= q1_fb) ]
higher_fb= lower_fb[lower_fb['movie_facebook_likes']<= q2_fb]

#filter imdb
q1_im= df['imdb_score'].quantile(0.05)
q2_im= df['imdb_score'].quantile(0.95)
print "q1 im =", q1_im
print "q2 im=" ,q2_im

lower_im= higher_fb[higher_fb['imdb_score'] >= q1_im]
higher_im= lower_im[lower_im['imdb_score'] <= q2_im ]

#filter budget
q1_b=df['budget'].quantile(0.05)
q2_b=df['budget'].quantile(0.95)
print "q1 b =", q1_b
print "q2 b=" ,q2_b

lower_b=higher_im[higher_im['budget'] >= q1_b]
higher_b= lower_b[lower_b['budget'] <= q2_b]

#rename 
filtered=higher_b

fb_likes=filtered.movie_facebook_likes
imdb=filtered.imdb_score
budget=filtered.budget
print "FB LIKES STATISTICS......."
print "Average= ", fb_likes.mean()
print "SD= ", fb_likes.std()

print "\n IMDB STATISTICS......."
print "Average= ", imdb.mean()
print "SD= ", imdb.std()


print " \n BUDGET STATISTICS......."
print "Average= ", budget.mean()
print "SD= ", budget.std()

#filtered.to_csv("movies_needCluster.csv", sep=',')
