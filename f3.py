# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:15:10 2023

@author: DEEP
"""
#Debug MF Model
# Import necessary libraries
import torch
import torch.nn as nn
import f1
# Debugging the matrix factorization model
print(f1.df_t_e)

# Define model parameters
num_users = 7
num_items = 4
emb_size = 3

# Create user and item embeddings
user_emb = nn.Embedding(num_users, emb_size)
item_emb = nn.Embedding(num_items, emb_size)

# Get user and item IDs
users = torch.LongTensor(f1.df_t_e.userId.values)
items = torch.LongTensor(f1.df_t_e.movieId.values)

# Look up user and item embeddings
U = user_emb(users)
V = item_emb(items)

# Calculate dot products per row
dot_products = (U * V).sum(1)

# Printing the results
print(U)
print(U * V)
print(dot_products)

