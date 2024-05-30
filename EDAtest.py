# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:20:56 2023

@author: DEEP
"""
#importing all necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Load the data
PATH = Path("D:/SEEM/winter sem/SE/Code_assign/Recom_sys/Data/ml-latest-small")

ratings = pd.read_csv(PATH / "ratings.csv")
movies = pd.read_csv(PATH / "movies.csv")


# Combine the ratings and movies dataframes
df = ratings.merge(movies, on='movieId')

# Drop the timestamp column
df.drop('timestamp', axis=1, inplace=True)

# Find the most popular genres
genres = {}
for genre in df['genres']:
    words = genre.split('|')
    for word in words:
        genres[word] = genres.get(word, 0) + 1

# Get the top 10 most popular genres
top_10_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]
print("The top ten generes are:",top_10_genres)

avg_ratings = df.groupby('title')['rating'].mean()
num_ratings = df.groupby('title')['rating'].count()

df_ratings = pd.DataFrame({'mean ratings': avg_ratings, 'total ratings': num_ratings})


df_ratings_sorted = df_ratings.sort_values('total ratings', ascending=False)


top_10_movies_by_ratings = df_ratings_sorted.head(10)


top_10_movies_by_mean_ratings = df_ratings_sorted.sort_values('mean ratings', ascending=False).head(10)


num_users_with_5_rating = len(df_ratings[df_ratings['mean ratings'] == 5])


num_single_users_with_5_rating = len(df_ratings[(df_ratings['mean ratings'] == 5) & (df_ratings['total ratings'] == 1)])

#Visualization of the data extracted
plt.figure(figsize=(8, 4))
sns.distplot(df_ratings['total ratings'], bins=20, color='b')
plt.xlabel('Total Number of Ratings')
plt.ylabel('Probability')
plt.show()

plt.figure(figsize=(8, 4))
sns.distplot(df_ratings['mean ratings'], bins=30, color='b')
plt.xlabel('Mean Ratings')
plt.ylabel('Probability')
plt.show()

sns.jointplot(x='mean ratings', y='total ratings', data=df_ratings)


#printing the Extracted data
print("Top 10 Movies by Ratings:",top_10_movies_by_ratings)
print("Top 10 Movies by Mean Ratings:",top_10_movies_by_mean_ratings)
print("Total Number of Users who gave Rating of 5.0:", num_users_with_5_rating)
print("Total Number of Single Users who gave Rating of 5.0:", num_single_users_with_5_rating)
