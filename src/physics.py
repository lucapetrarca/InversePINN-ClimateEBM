import torch

#Se calcula el residuo de la ecuación diferencial de Budyko-Sellers

#Q_solar = Radiación solar incidente media (constante)

def calculate_physics_loss(model, x, Q_solar=340.0):
    #Se obtiene la predicción de temperatura
    T = model(x)
    
    #Se calculan la primera y segunda derivada de T respecto a x
    dT_dx = torch.autograd.grad(
        T, x, 
        grad_outputs=torch.ones_like(T),
        create_graph=True, #Se crea el grafo para poder derivar nuevamente
        retain_graph=True
    )[0]
    
    d2T_dx2 = torch.autograd.grad(
        dT_dx, x, 
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
        retain_graph=True
    )[0]
    
    #Ecuación de balance de energía
    
    #Radiación Solar Entrante -albedo constante-
    albedo = 0.3

    #Aproximación: Q(x) = Q_solar * (1 - 0.482 * x^2)
    Q_in = Q_solar * (1.0 - 0.482 * (x**2)) * (1.0 - albedo)
    
    #Radiación Infrarroja Saliente (R_out = A + B*T), usando parámetros descubiertos
    R_out = model.A_out + model.B_out * T
    
    #Transporte de Calor (Difusión)
    #D * [ -2x * dT/dx + (1-x^2) * d2T/dx2 ]
    transporte_calor = model.D * ( -2.0 * x * dT_dx + (1.0 - x**2) * d2T_dx2 )
    
    #Residuo
    f_residuo = transporte_calor + Q_in - R_out
    
    #La loss física es el Error Cuadrático Medio del residuo
    loss_physics = torch.mean(f_residuo**2)
    
    return loss_physics