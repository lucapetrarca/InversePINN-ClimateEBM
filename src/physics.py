import torch
import torch.nn.functional as F

#Se calcula el residuo de la ecuación diferencial de Budyko-Sellers

def calculate_physics_loss(model, x, Q_solar=340.0):
    #El modelo devuelve la Temperatura y el Campo de Difusividad
    T, D_efectivo = model(x)
    
    #Derivadas de la Temperatura
    dT_dx = torch.autograd.grad(
        T, x, 
        grad_outputs=torch.ones_like(T),
        create_graph=True,
        retain_graph=True
    )[0]
    
    d2T_dx2 = torch.autograd.grad(
        dT_dx, x, 
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
        retain_graph=True
    )[0]
    
    #Derivada del campo D(x)
    dD_dx = torch.autograd.grad(
        D_efectivo, x,
        grad_outputs=torch.ones_like(D_efectivo),
        create_graph=True,
        retain_graph=True
    )[0]
    
    #Radiación Solar Entrante - Albedo Dinámico
    albedo = 0.5 - 0.2 * torch.tanh(0.1 * (T - 263.15))
    Q_in = Q_solar * (1.0 - 0.482 * (x**2)) * (1.0 - albedo)
    
    T_celsius = T - 273.15
    
    #B > 0
    B_efectivo = F.softplus(model.B_out)
    
    #Ecuación Asimétrica.
    R_out = model.A_out + B_efectivo * T_celsius + model.C_out * x

    #Transporte de Calor (Difusión) con regla del producto completa:
    transporte_calor = (
        dD_dx * (1.0 - x**2) * dT_dx + 
        D_efectivo * (-2.0 * x) * dT_dx + 
        D_efectivo * (1.0 - x**2) * d2T_dx2
    )
    
    f_residuo = transporte_calor + Q_in - R_out
    loss_physics = torch.mean(f_residuo**2)
    
    return loss_physics