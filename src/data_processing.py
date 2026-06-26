import pandas as pd
import torch
import numpy as np

def get_training_data(csv_path='../data/temp_latitudinal.csv'):
    df = pd.read_csv(csv_path)
    x_data = torch.tensor(df['x_seno'].values, dtype=torch.float32).view(-1, 1)
    T_data = torch.tensor(df['temperatura_K'].values, dtype=torch.float32).view(-1, 1)
    return x_data, T_data

def get_collocation_points(n_points=200):
    #Se generan collocation points donde se debe cumplir la ecuación diferencial
    x_physics = torch.linspace(-1.0, 1.0, n_points, dtype=torch.float32).view(-1, 1)

    #A x_physics se le exige esto para poder correr physics.py
    x_physics.requires_grad = True 
    
    return x_physics