# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:49:23 2023

@author: DEEP
"""

# Import necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np

# Set the data path
PATH = Path("D:/SEEM/winter sem/SE/Code_assign/Recom_sys/Data/ml-latest-small")

# List directory contents
list(PATH.iterdir())

# Read data from ratings.csv
data = pd.read_csv(PATH / "ratings.csv")
print(data.shape)

# Split data into training and validation
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()
print(train.shape)

# Define a function to encode a column
def encode_column(col, train_col=None):
    if train_col is not None:
        unique_vals = train_col.unique()
    else:
        unique_vals = col.unique()
    value_to_index = {value: index for index, value in enumerate(unique_vals)}
    encoded_col = np.array([value_to_index.get(x, -1) for x in col])
    return value_to_index, encoded_col, len(unique_vals)

# Define a function to encode the data
def encode_data(df, train=None):
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = train[col_name] if train is not None else df[col_name]
        value_to_index, encoded_col, _ = encode_column(df[col_name], train_col)
        df[col_name] = encoded_col
        df = df[df[col_name] >= 0]
    return df

# Load additional data for testing
LOCAL_PATH = Path("D:/SEEM/winter sem/SE/Code_assign/Recom_sys/Data")
df_t = pd.read_csv(LOCAL_PATH / "tiny_training2.csv")
df_v = pd.read_csv(LOCAL_PATH / "tiny_val2.csv")
print(df_t)

# Encode the data
df_t_e = encode_data(df_t)
df_v_e = encode_data(df_v, df_t)
df_v_e
print(df_t_e)


