import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F

def get_training_data(csv_path='../data/temp_latitudinal.csv'):
    df = pd.read_csv(csv_path)
    x_data = torch.tensor(df['x_seno'].values, dtype=torch.float32).view(-1, 1)
    T_data = torch.tensor(df['temperatura_K'].values, dtype=torch.float32).view(-1, 1)
    return x_data, T_data

def get_collocation_points(n_points=200):
    x_physics = torch.linspace(-1.0, 1.0, n_points, dtype=torch.float32).view(-1, 1)
    x_physics.requires_grad = True 
    return x_physics

#RAR Evalúa la ecuación diferencial en una grilla MUY densa, encuentra dónde el residuo es mayor y agrega esos puntos difíciles
def get_rar_collocation_points(model, current_x_physics, n_new_points=20, Q_solar=340.0):
    x_test = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float32).view(-1, 1)
    x_test.requires_grad_(True)
    
    T, D_efectivo = model(x_test)
    
    dT_dx = torch.autograd.grad(T, x_test, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x_test, grad_outputs=torch.ones_like(dT_dx), create_graph=True, retain_graph=True)[0]
    
    dD_dx = torch.autograd.grad(D_efectivo, x_test, grad_outputs=torch.ones_like(D_efectivo), create_graph=True, retain_graph=True)[0]
    
    albedo = 0.5 - 0.2 * torch.tanh(0.1 * (T - 263.15))
    Q_in = Q_solar * (1.0 - 0.482 * (x_test**2)) * (1.0 - albedo)
    T_celsius = T - 273.15
    
    #B positivo
    B_efectivo = F.softplus(model.B_out)
    R_out = model.A_out + B_efectivo * T_celsius + model.C_out * x_test
    
    transporte_calor = (
        dD_dx * (1.0 - x_test**2) * dT_dx + 
        D_efectivo * (-2.0 * x_test) * dT_dx + 
        D_efectivo * (1.0 - x_test**2) * d2T_dx2
    )
    
    f_residuo = transporte_calor + Q_in - R_out
    res_abs = torch.abs(f_residuo).detach()
    
    _, top_indices = torch.topk(res_abs.flatten(), n_new_points)
    x_new = x_test[top_indices].clone().detach()
    
    x_combined = torch.cat([current_x_physics.detach(), x_new], dim=0)
    x_combined, _ = torch.sort(x_combined, dim=0)
    x_combined.requires_grad_(True)
    
    return x_combined