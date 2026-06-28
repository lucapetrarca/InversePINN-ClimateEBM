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

#Muestreo importante

def get_rar_collocation_points(model, current_x_physics, n_new_points=20, Q_solar=340.0):
    #Se crea una grilla densa de prueba
    x_test = torch.linspace(-1.0, 1.0, 3000, dtype=torch.float32).view(-1, 1)
    x_test.requires_grad_(True)
    
    #Se calcula la pérdida física punto por punto
    from src.physics import calculate_physics_loss
    
    T = model(x_test)
    dT_dx = torch.autograd.grad(T, x_test, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x_test, grad_outputs=torch.ones_like(dT_dx), create_graph=True, retain_graph=True)[0]
    
    albedo = 0.5 - 0.2 * torch.tanh(0.1 * (T - 263.15))
    Q_in = Q_solar * (1.0 - 0.482 * (x_test**2)) * (1.0 - albedo)
    T_celsius = T - 273.15
    R_out = model.A_out + model.B_out * T_celsius + model.C_out * x_test
    transporte_calor = model.D * ( -2.0 * x_test * dT_dx + (1.0 - x_test**2) * d2T_dx2 )
    f_residuo = transporte_calor + Q_in - R_out
    
    res_abs = torch.abs(f_residuo).detach()
    
    #Se encuentran los índices de los 'n_new_points' con mayor error
    _, top_indices = torch.topk(res_abs.flatten(), n_new_points)
    
    x_new = x_test[top_indices].clone().detach()
    
    x_combined = torch.cat([current_x_physics.detach(), x_new], dim=0)
    x_combined, _ = torch.sort(x_combined, dim=0)
    
    x_combined.requires_grad_(True)
    
    return x_combined