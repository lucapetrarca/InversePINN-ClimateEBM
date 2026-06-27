import torch

def calculate_physics_loss(model, x, Q_solar=340.0):
    #Se obtiene la predicción de temperatura (Ya viene en Kelvin)
    T = model(x)
    
    #Se calculan la primera y segunda derivada de T respecto a x
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
    
    #Radiación Solar Entrante - Albedo Dinámico
    albedo = 0.5 - 0.2 * torch.tanh(0.1 * (T - 263.15))
    Q_in = Q_solar * (1.0 - 0.482 * (x**2)) * (1.0 - albedo)
    
    T_celsius = T - 273.15
    
    #C*x compensa la diferencia térmica entre el Polo Norte (x=1) y el Polo Sur (x=-1)
    R_out = model.A_out + model.B_out * T_celsius + model.C_out * x

    #Transporte de Calor (Difusión)
    transporte_calor = model.D * ( -2.0 * x * dT_dx + (1.0 - x**2) * d2T_dx2 )
    
    #Residuo
    f_residuo = transporte_calor + Q_in - R_out
    
    loss_physics = torch.mean(f_residuo**2)
    
    return loss_physics