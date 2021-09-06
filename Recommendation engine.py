
import pandas as pd 
import numpy as np 
df1=pd.read_csv(r"C:\Users\Pranita\Documents\latest dataset\credits.csv")
df2=pd.read_csv(r"C:\Users\Pranita\Documents\latest dataset\movies.csv")

df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

sns.set_style('dark')
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
#df2['vote_count'].hist(bins=50)
plt.hist(df2['vote_count'],bins=50)
plt.xlabel("Ratings")
plt.ylabel("Number of ratings")
#plt.hist(exp_data, bins=21, align='left', color='b', edgecolor='red',linewidth=1)
plt.show()

sns.set_style('dark')
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
#df2['vote_average'].hist(bins=50)
plt.hist(df2['vote_average'],bins=50)
plt.xlabel("Average Ratings for every movie")
plt.ylabel("Number of Ratings")
plt.show()

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
scat=sns.jointplot(x='vote_average', y='vote_count', data=df2, alpha=0.4)
plt.show()


df2['overview'].head(5)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


def recommendations(title, cosine_sim=cosine_sim):
   
    id = indices[title]
    sim_scores = list(enumerate(cosine_sim[id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]


movie=input("enter the name of the movie you need recommendations for :\n")
print(recommendations(movie))






