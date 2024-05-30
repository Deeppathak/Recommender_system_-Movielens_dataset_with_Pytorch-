# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:21:13 2023

@author: DEEP
"""

# Handle table-like data and matrices :
import numpy as np
import pandas as pd
import math 
import itertools
from pathlib import Path


# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#import missingno as msno


# Configure visualisations
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)


PATH = Path("D:\SEEM\winter sem\SE\Data\ml-latest-small")
ratings = pd.read_csv(PATH / "ratings.csv")
movies = pd.read_csv(PATH / "movies.csv")
df_r = ratings.copy()
df_m = movies.copy()

ratings.head()

ratings.shape

ratings.describe()

ratings.drop(['timestamp'], axis=1, inplace=True)
ratings.head()
movies.head()
print('Shape: ', movies.shape, '\n')
movies.info()

df_combined = pd.merge(ratings, movies, on = 'movieId')
df_combined.head()
df_combined.shape

# Create a function to find genres in the dataset

genres = {} # create a dictionary to store different genre values

def find_genres():
    for genre in movies['genres']:
        words = genre.split('|')
        for word in words:
            genres[word] = genres.get(word, 0) + 1
            
find_genres()

genres
genres['None'] = genres.pop('(no genres listed)')



## Heavily Rated Movies
df_n_ratings = pd.DataFrame(df_combined.groupby('title')['rating'].mean())
df_n_ratings['total ratings'] = pd.DataFrame(df_combined.groupby('title')['rating'].count())
df_n_ratings.rename(columns = {'rating': 'mean ratings'}, inplace=True)

df_n_ratings.sort_values('total ratings', ascending=False).head(10)

plt.figure(figsize=(8,4))
sns.distplot(df_n_ratings['total ratings'], bins=20)
plt.xlabel('Total Number of Ratings')
plt.ylabel('Probability')
plt.show()


df_n_ratings.sort_values('mean ratings', ascending=False).head(10)


print('Total no of users that gave rating of 5.0 : ', len(df_n_ratings.loc[df_n_ratings['mean ratings'] == 5]), '\n')
print('Total no of Individual users that gave rating of 5.0 : ', len(df_n_ratings.loc[(df_n_ratings['mean ratings'] == 5) 
                                                                           & (df_n_ratings['total ratings'] == 1)]))


plt.figure(figsize=(8,4))
sns.distplot(df_n_ratings['mean ratings'], bins=30)
plt.xlabel('Mean Ratings')
plt.ylabel('Probability')
plt.show()


#Mean Ratings vs Total Number of Ratings

sns.jointplot(x = 'mean ratings', y = 'total ratings', data = df_n_ratings )