# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:59:18 2023

@author: DEEP
"""
#Imbedding the layer
# Import  necessary libraries
import torch
import torch.nn as nn
from f1 import encode_data 
from f1 import train
from f1 import val
from pathlib import Path
import pandas as pd
from f1 import PATH
# Matrix factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u * v).sum(1)

# Encoding the train and validation data
df_train = encode_data(train)
df_val = encode_data(val,train)


print(df_train.head(10))
print(df_val.head(10))

# Filter data for user x
user_data = df_val[df_val['userId'] == 16]  # Assuming userId starts from 0

# Sort movies by rating in descending order for user x
top_rated_movies_user = user_data.sort_values(by='rating', ascending=False)

# Merge with movie genres
movies_data = pd.read_csv(PATH / "movies.csv")
top_rated_movies_user_with_genres = pd.merge(top_rated_movies_user, movies_data, on='movieId')
top_rated_movies_user_with_genres=top_rated_movies_user_with_genres.head(100)
# Display top-rated movies for user 5 along with genres
print("Top Rated Movies for User x:")
print(top_rated_movies_user_with_genres[[ 'title', 'genres', 'rating']])