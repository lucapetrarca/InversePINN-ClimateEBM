import torch
import torch.nn as nn
import torch.nn.functional as F

class ClimateInversePINN(nn.Module):
    def __init__(self, hidden_layers=5, neurons_per_layer=64):
        super(ClimateInversePINN, self).__init__()
        #Input: x (1D)
        #Output: T_norm (1D) y D_norm (1D)
        layers = []
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(neurons_per_layer, 2))
        
        self.net = nn.Sequential(*layers)
        
        #Parámetros físicos constantes a descubrir
        self.A_out = nn.Parameter(torch.tensor([200.0], dtype=torch.float32))
        self.B_out = nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
        self.C_out = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
    def forward(self, x):
        #La red devuelve un tensor de tamaño [N, 2]
        output = self.net(x)
        
        #Separación de las dos predicciones
        T_norm = output[:, 0:1] #Primera columna: Temperatura
        D_raw = output[:, 1:2]  #Segunda columna: Difusividad cruda
        
        #Escalado físico
        #Temperatura: Rango biológico (alrededor de 273 K)
        T_kelvin = T_norm * 50.0 + 273.15
        
        #Difusividad: D(x) Debe ser positiva
        D_efectivo = F.softplus(D_raw)
        
        return T_kelvin, D_efectivo