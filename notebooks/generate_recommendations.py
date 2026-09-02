import numpy as np
import pandas as pd

movie = pd.read_csv("tmdb_5000_movies.csv")
credit = pd.read_csv("tmdb_5000_credits.csv")

movie.head()

credit.head(1)['cast'].values
credit.head(1)['crew'].values
#movie.merge(credit ,on='title').shape
movie = movie.merge(credit , on='title')
movie.head(1)

#required coulmns -> geners , id ,keywords , overview , cast , crew , title

movie = movie[['movie_id','title','overview','genres','keywords','cast','crew']]
movie.head()
movie.isnull().sum()
movie.dropna(inplace=True)

movie.duplicated().sum()
movie.iloc[0].genres
#output is list of dictornaries
#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#need to convert - ['Action' ,'Adventure']
import ast
#convert string to list
def convert(obj):
    lis = []
    for i in ast.literal_eval(obj):
        lis.append(i['name'])
    return lis
movie['genres'] = movie['genres'].apply(convert)

movie['keywords']=movie['keywords'].apply(convert)
movie.head(1)
movie['cast'][0]
def convert3(obj):
    lis = []
    counter =0
    for i in ast.literal_eval(obj):
        if counter !=3 :
           lis.append(i['name'])
           counter+=1
        else :
           break
    return lis
movie['cast']=movie['cast'].apply(convert3)

movie['crew'][0]
def director(obj):
    lis = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
           lis.append(i['name'])
           break;
    return lis
movie['crew'] = movie['crew'].apply(director)
movie['crew'][0]
movie['overview'][0]
movie['overview'] = movie['overview'].apply(lambda x : x.split())
movie['genres']=movie['genres'].apply(lambda x : [i.replace(" ","") for i in x])
movie['cast']=movie['cast'].apply(lambda x : [i.replace(" ","") for i in x])
movie['crew']=movie['crew'].apply(lambda x : [i.replace(" ","") for i in x])
movie['keywords']=movie['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
movie['tags'] = movie['overview']+movie['genres']+movie['cast']+movie['crew']
movie.head(1)
new_df = movie[['movie_id','title','tags']]

new_df.head(2)
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head(2)
new_df['tags']= new_df['tags'].apply(lambda x : x.lower())
# Here we are performing vectorization, here we do not use stop words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']) # Keep it sparse!

# now we apply steming
import nltk 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)

# We do NOT precompute cosine similarity anymore to save memory
import pickle

# 1. Export the dataframe as a dictionary
with open('movies_dict.pkl', 'wb') as f:
    pickle.dump(new_df.to_dict(), f)

# 2. Export the sparse vectors matrix
with open('vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)

print("Files generated successfully! (Optimized Sparse Matrix)")
