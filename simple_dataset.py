# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:47:55 2023

@author: DEEP
"""


import torch


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, inputs, outputs):
        
        self.ins = inputs
        self.outs= outputs
        
        
    def __len__(self):
        return len(self.ins)

    def __getitem__(self, index):
        return self.ins[index], self.outs[index]
    
    
    
    
#%%

inputs = torch.rand(200,16)
outputs = torch.rand(200)

dataset = Dataset(inputs, outputs)

dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=4, 
                                         shuffle=True, 
                                         num_workers=0)


for ins, outs in dataloader:
    
    mres = odel(ins, outs)
    
    loss = res-outs
    
    print(outs)


user_id = 16
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
    PATH = Path("D:\SEEM\winter sem\SE\Data\ml-latest-small")
    movies_data = pd.read_csv(PATH/"movies.csv")

    # Assuming your 'movies.csv' file contains columns 'movieId' and 'title'
    df_movies = movies_data[['movieId', 'title']]

    # Now, you have the DataFrame 'df_movies' with movie IDs and titles

    
    # Get movie names from the DataFrame
    top_movie_names = df_movies[df_movies['movieId'].isin(top_movie_ids)]['title'].tolist()
    
    return top_movie_names


 # Replace with the user ID for whom you want to make recommendations
top_movies = recommend_top_movies(user_id, model, num_items, top_n=6)

# Print the top recommended movies for the user
print("Top 5 recommended movies for user", user_id, ":")
for i, movie_id in enumerate(top_movies):
    print(f"Rank {i+1}: {movie_id}")


