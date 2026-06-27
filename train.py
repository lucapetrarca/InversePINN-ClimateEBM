import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import ClimateInversePINN
from src.physics import calculate_physics_loss
from src.data_processing import get_training_data, get_collocation_points

def train_inverse_pinn(epochs_adam=5000, lambda_physics_max=1.0):
    print("Iniciando entrenamiento de la Inverse PINN...")
    
    #Se cargan los datos y puntos físicos
    try:
        x_data, T_data = get_training_data('data/temp_latitudinal.csv')
        print(f"Datos reales cargados: {x_data.shape[0]} latitudes.")
    except Exception as e:
        print("Error cargando el CSV. Asegurate de haber corrido data_loader.ipynb primero.")
        return None, None
        
    x_physics = get_collocation_points(n_points=200)
    
    #Los datos también minimizan loss física
    x_data.requires_grad_(True)
    
    #Se inicializa el modelo
    model = ClimateInversePINN(hidden_layers=4, neurons_per_layer=32)
    
    #Fase 1: ADAM
    #Búsqueda rápida
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    
    history = {'loss_total': [], 'loss_data': [], 'loss_physics': [], 
               'D': [], 'A': [], 'B': []}
    
    print("\n--- FASE 1: Entrenando con Adam ---")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        
        #Curriculum learning
        lambda_val = lambda_physics_max * (epoch / epochs_adam)
        
        #Loss Data
        T_pred = model(x_data)
        loss_data = torch.mean((T_pred - T_data)**2)
        
        #Loss Física
        loss_physics_colloc = calculate_physics_loss(model, x_physics)
        loss_physics_data = calculate_physics_loss(model, x_data)
        loss_physics = loss_physics_colloc + loss_physics_data
        
        #Loss Total
        loss_total = loss_data + lambda_val * loss_physics
        
        #Backpropagation
        loss_total.backward()
        optimizer_adam.step()
        
        #Se guardan valores cada 100 epochs
        if epoch % 100 == 0:
            D_val = model.D.item()
            A_val = model.A_out.item()
            B_val = model.B_out.item()
            
            history['loss_total'].append(loss_total.item())
            history['D'].append(D_val)
            history['loss_data'].append(loss_data.item())
            history['loss_physics'].append(loss_physics.item())
            history['A'].append(A_val)
            history['B'].append(B_val)
            
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch:04d} | L_Total: {loss_total.item():.2f} | L_Data: {loss_data.item():.2f} | L_Phys: {loss_physics.item():.2f}")

    #Fase 2: L-BFGS
    #Lidia con los mínimos donde Adam se queda estancado.
    print("\n--- FASE 2: Entrenando con L-BFGS (Ajuste Fino) ---")
    
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=0.1, 
        max_iter=2000, 
        max_eval=2000, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=100
    )
    
    #L-BFGS requiere una función 'closure' que re-evalúe todo el grafo
    def closure():
        optimizer_lbfgs.zero_grad()
        
        T_pred = model(x_data)
        loss_data = torch.mean((T_pred - T_data)**2)
        
        loss_physics_colloc = calculate_physics_loss(model, x_physics)
        loss_physics_data = calculate_physics_loss(model, x_data)
        loss_physics = loss_physics_colloc + loss_physics_data
        
        #En la fase L-BFGS, se usa el lambda máximo completo
        loss_total = loss_data + lambda_physics_max * loss_physics
        loss_total.backward()
        return loss_total

    optimizer_lbfgs.step(closure)
    
    final_loss = closure()
    print(f"L-BFGS Final Loss: {final_loss.item():.4f}")

    print("\n¡Entrenamiento finalizado!")
    print(f"Parámetros finales descubiertos: D={model.D.item():.4f}, A={model.A_out.item():.2f}, B={model.B_out.item():.4f}")
    
    #Validación visual
    print("\nGenerando gráfico de validación...")
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 200).view(-1, 1)
        T_plot = model(x_plot)
        
    plt.figure(figsize=(9, 6))
    plt.plot(x_data.detach().numpy(), T_data.detach().numpy() - 273.15, 'ro', label='Datos Reales NOAA (CSV)')
    plt.plot(x_plot.detach().numpy(), T_plot.detach().numpy() - 273.15, 'b-', linewidth=2.5, label='Curva Continua Inverse PINN')
    
    plt.xlabel(r'Variable espacial $x = \sin(latitud)$')
    plt.ylabel('Temperatura (°C)')
    plt.title('Validación del Equilibrio Climático Topológico (Adam + L-BFGS)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return model, history

if __name__ == "__main__":
    #Se corre el entrenamiento
    trained_model, training_history = train_inverse_pinn(epochs_adam=3000, lambda_physics_max=1.0)