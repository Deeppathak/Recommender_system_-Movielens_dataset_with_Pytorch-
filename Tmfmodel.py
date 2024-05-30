# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:52:12 2023

@author: DEEP
"""

from f1 import encode_data 
import f2
import f1
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from f2 import MatrixFactorization
from f2 import df_train
from f2 import df_val

def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    
    for epoch in range(epochs):
        users = torch.LongTensor(df_train.userId.values)
        items = torch.LongTensor(df_train.movieId.values)
        ratings = torch.FloatTensor(df_train.rating.values)
        
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

    test_loss(model, unsqueeze)

def test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_val.userId.values)
    items = torch.LongTensor(df_val.movieId.values)
    ratings = torch.FloatTensor(df_val.rating.values)
    
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    
    print(f"Test Loss: {loss.item():.4f}")

# Get the number of unique users and items
num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
print(f"Number of Users: {num_users}, Number of Items: {num_items}")

# Create the MF model
model = MatrixFactorization(num_users, num_items, emb_size=100)

# Training epochs
train_epocs(model, epochs=10, lr=0.1)
train_epocs(model, epochs=15, lr=0.01)
train_epocs(model, epochs=15, lr=0.01)


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
    print("Top  recommended movies for user", user_id, ":")
    for i, row in top_movies_info.iterrows():
        print(f"Rank {i+1}: Movie ID {row['movieId']}, Title: {row['title']}, Genres: {row['genres']}")

# Replace with the user ID for whom you want to make recommendations
user_id = 16
recommend_top_movies(user_id, model, num_items, top_n=6)
