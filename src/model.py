import torch
import torch.nn as nn

class ClimateInversePINN(nn.Module):
    def __init__(self, hidden_layers=4, neurons_per_layer=32):
        super(ClimateInversePINN, self).__init__()
        #Input: x (seno de la latitud, 1D) -> Output: T (Temperatura en Kelvin, 1D)
        layers = []
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.net = nn.Sequential(*layers)
        
        #Parámetros Físicos a descubrir
        
        #Coeficiente "D" de difusión de calor
        self.D = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        
        #Constantes de radiación saliente (R_out = A + B*T)
        self.A_out = nn.Parameter(torch.tensor([200.0], dtype=torch.float32))
        self.B_out = nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
        
    def forward(self, x):
        out_norm = self.net(x)
        T_kelvin = out_norm * 50.0 + 273.15 #Se fuerza el arranque de la temperatura en el rango de temperaturas biológicas de la tierra
        
        return T_kelvin