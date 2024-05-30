# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:02:11 2023

@author: DEEP
"""

from f1 import encode_data 
from Tmfmodel import num_items, num_users, train_epocs
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=300):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
    
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U * V).sum(1) + b_u + b_v

model = MF_bias(num_users, num_items, emb_size=300)

train_epocs(model, epochs=10, lr=0.05, wd=0.001)
train_epocs(model, epochs=10, lr=0.01, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=0.0001)
train_epocs(model, epochs=10, lr=0.001, wd=1e-10)
train_epocs(model, epochs=10, lr=0.001, wd=1e-10)
train_epocs(model, epochs=10, lr=0.001, wd=1e-10)
train_epocs(model, epochs=10, lr=0.001, wd=1e-10)



def recommend_top_movies(user_id, model, num_items, top_n=5):
    # Create a list of movie IDs for all items
    item_ids = torch.arange(num_items)
    
    # Repeat the user ID for all items
    user_ids = torch.full((num_items,), user_id, dtype=torch.long)
    
    # Get predictions for all items for the given user
    predictions = model(user_ids, item_ids)
    
    # Sort the predictions in descending order to get the top recommendations
    top_indices = predictions.view(-1).argsort(descending=True)[:top_n]
    
    # Retrieve the top movie IDs
    top_movie_ids = item_ids[top_indices]
    
    # Load movie data
    PATH = Path("D:\SEEM\winter sem\SE\Data\ml-latest-small")
    movies_data = pd.read_csv(PATH/"movies.csv")
    
    # Merge with movies_data to get genres
    top_movies_info = pd.merge(pd.DataFrame(top_movie_ids, columns=['movieId']), movies_data, on='movieId')

    # Display top recommended movies with genres
    print("Top recommended movies for user", user_id, ":")
    for i, row in top_movies_info.iterrows():
        print(f"Rank {i+1}: Movie ID {row['movieId']}, Title: {row['title']}, Genres: {row['genres']}")

# Replace with the user ID for whom you want to make recommendations
user_id = 16
recommend_top_movies(user_id, model, num_items, top_n=6)