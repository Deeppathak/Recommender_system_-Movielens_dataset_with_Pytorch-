# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:20:56 2023

@author: DEEP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PATH = Path("D:/SEEM/winter sem/SE/Code_assign/Recom_sys/Data/ml-latest-small")

ratings = pd.read_csv(PATH / "ratings.csv")
movies = pd.read_csv(PATH / "movies.csv")

ratings.drop(['timestamp'], axis=1, inplace=True)

df_combined = pd.merge(ratings, movies, on='movieId')

genres = {}

def find_genres():
    for genre in movies['genres']:
        words = genre.split('|')
        for word in words:
            genres[word] = genres.get(word, 0) + 1

find_genres()

genres['None'] = genres.pop('(no genres listed)')
first_10_items = list(genres.items())[:10]

print(first_10_items)
df_n_ratings = pd.DataFrame(df_combined.groupby('title')['rating'].mean())
df_n_ratings['total ratings'] = pd.DataFrame(df_combined.groupby('title')['rating'].count())
df_n_ratings.rename(columns={'rating': 'mean ratings'}, inplace=True)
top_movies_by_ratings = df_n_ratings.sort_values('total ratings', ascending=False).head(10)
print("Top Movies by ratings are:",top_movies_by_ratings)
plt.figure(figsize=(8, 4))
sns.distplot(df_n_ratings['total ratings'], bins=20,color="b")
plt.xlabel('Total Number of Ratings')
plt.ylabel('Probability')
plt.show()

top_movies_by_mean_ratings = df_n_ratings.sort_values('mean ratings', ascending=False).head(10)
print("Top Movies by mean ratings are:",top_movies_by_mean_ratings)



print('Total number of users that gave a rating of 5.0: ', len(df_n_ratings.loc[df_n_ratings['mean ratings'] == 5]))
print('Total number of individual users that gave a rating of 5.0: ',
      len(df_n_ratings.loc[(df_n_ratings['mean ratings'] == 5) & (df_n_ratings['total ratings'] == 1)]))

plt.figure(figsize=(8, 4))
sns.distplot(df_n_ratings['mean ratings'], bins=30,color="b")
plt.xlabel('Mean Ratings')
plt.ylabel('Probability')
plt.show()

sns.jointplot(x='mean ratings', y='total ratings', data=df_n_ratings)

top_movies_by_ratings
