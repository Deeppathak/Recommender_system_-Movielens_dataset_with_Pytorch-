# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:31:01 2023

@author: DEEP
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from f1 import encode_data
import f2
import f1
from Tmfmodel import num_items
from Tmfmodel import num_users
from Tmfmodel import train_epocs
from pathlib import Path
import pandas as pd
import numpy as np


# Load ratings data for all users
PATH = Path("D:\SEEM\winter sem\SE\Data\ml-latest-small")
ratings_df = pd.read_csv(PATH/"ratings.csv")
ratings = ratings_df.values.astype(float)

class CFNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=35, n_hidden=7):
        super().__init__()

        # Embedding layers for users and items
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        # Fully connected layers
        self.lin1 = nn.Linear(emb_size * 2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, 1)

        # Dropout layer for regularization
        self.drop1 = nn.Dropout(0.1)

    def forward(self, user_id, item_id):
        # Get user and item embeddings
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)

        # Concatenate user and item embeddings
        cat_emb = torch.cat([user_emb, item_emb], dim=1)

        # Apply a ReLU activation function
        x = F.relu(cat_emb)

        # Apply dropout regularization
        x = self.drop1(x)

        # Apply a series of fully connected layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)

        # Apply a final output layer to predict the rating
        prediction = self.lin3(x)

        return prediction
# Create the Collaborative filter model
model = CFNet(num_users, num_items, emb_size=35) 

# Train the model with different hyperparameters

train_epocs(model, epochs=15, lr=0.01, wd=0.01, unsqueeze=True) 
train_epocs(model, epochs=15, lr=0.01, wd=0.001, unsqueeze=True) 
train_epocs(model, epochs=10, lr=0.01, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.01, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.01, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.01, wd=1e-6, unsqueeze=True)
train_epocs(model, epochs=100, lr=0.01, wd=1e-10, unsqueeze=True)
train_epocs(model, epochs=10, lr=0.001, wd=1e-6, unsqueeze=True)


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
recommend_top_movies(user_id, model, num_items, top_n=10)

