import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.model import ClimateInversePINN
from src.physics import calculate_physics_loss
from src.data_processing import get_training_data, get_collocation_points, get_rar_collocation_points

def train_inverse_pinn(epochs_adam=5000, lambda_physics_max=0.5):
    print("Iniciando entrenamiento de la PINN con Field Discovery D(x) y Física Restringida")
    
    try:
        x_data, T_data = get_training_data('data/temp_latitudinal.csv')
        print(f"Datos reales cargados: {x_data.shape[0]} latitudes.")
    except Exception as e:
        print("Error cargando el CSV. Asegurate de haber corrido data_loader.ipynb primero.")
        return None, None
        
    x_physics = get_collocation_points(n_points=100)
    x_data.requires_grad_(True)
    
    model = ClimateInversePINN(hidden_layers=5, neurons_per_layer=64)
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    
    history = {'loss_total': [], 'loss_data': [], 'loss_physics': [], 
               'D_mean': [], 'A': [], 'B': [], 'C': []}
    
    print("\nFASE 1: Adam con RAR")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        
        lambda_val = lambda_physics_max * (epoch / epochs_adam)
        
        T_pred, _ = model(x_data)
        loss_data = torch.mean((T_pred - T_data)**2)
        
        loss_physics_colloc = calculate_physics_loss(model, x_physics)
        loss_physics_data = calculate_physics_loss(model, x_data)
        loss_physics = loss_physics_colloc + loss_physics_data
        
        loss_total = loss_data + lambda_val * loss_physics
        loss_total.backward()
        optimizer_adam.step()
        
        if epoch > 0 and epoch % 1000 == 0:
            x_physics = get_rar_collocation_points(model, x_physics, n_new_points=20)
            print(f"   [RAR] Agregados 20 nuevos puntos. Total puntos físicos: {x_physics.shape[0]}")
        
        if epoch % 100 == 0:
            _, D_pred_data = model(x_data)
            D_mean_val = torch.mean(D_pred_data).item()
            
            A_val = model.A_out.item()
            B_val = F.softplus(model.B_out).item()
            C_val = model.C_out.item()
            
            history['loss_total'].append(loss_total.item())
            history['D_mean'].append(D_mean_val)
            history['loss_data'].append(loss_data.item())
            history['loss_physics'].append(loss_physics.item())
            history['A'].append(A_val)
            history['B'].append(B_val)
            history['C'].append(C_val)
            
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch:04d} | L_Total: {loss_total.item():.2f} | L_Data: {loss_data.item():.2f} | L_Phys: {loss_physics.item():.2f}")

    print("\n--- FASE 2: Entrenando con L-BFGS ---")
    
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1.0,  
        max_iter=50000, 
        max_eval=50000, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=200,
        line_search_fn="strong_wolfe"
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        
        T_pred, _ = model(x_data)
        loss_data = torch.mean((T_pred - T_data)**2)
        
        loss_physics_colloc = calculate_physics_loss(model, x_physics)
        loss_physics_data = calculate_physics_loss(model, x_data)
        loss_physics = loss_physics_colloc + loss_physics_data
        
        loss_total = loss_data + lambda_physics_max * loss_physics
        loss_total.backward()
        return loss_total

    optimizer_lbfgs.step(closure)
    
    print("\nEntrenamiento finalizado")
    model.eval()
    
    T_pred_final, _ = model(x_data)
    final_L_data = torch.mean((T_pred_final - T_data)**2).item()
    final_L_phys_colloc = calculate_physics_loss(model, x_physics).item()
    final_L_phys_data = calculate_physics_loss(model, x_data).item()
    
    print("\n--- VALIDACIÓN CIENTÍFICA ---")
    print(f"1. Error Estadístico (L_Data MSE):        {final_L_data:.4f} °C^2")
    print(f"2. Error Físico en Sensores (L_Phys):     {final_L_phys_data:.4f}")
    print(f"3. Error de Generalización (L_Colloc):    {final_L_phys_colloc:.4f}")
    
    final_B = F.softplus(model.B_out).item()
    print(f"\nParámetros Constantes Descubiertos (100% Válidos):")
    print(f"A={model.A_out.item():.2f}, B={final_B:.4f}, C={model.C_out.item():.4f}")
    
    print("\nGenerando gráficos de validación...")
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 200).view(-1, 1)
        T_plot, D_plot = model(x_plot)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(x_data.detach().numpy(), T_data.detach().numpy() - 273.15, 'ro', label='Datos Reales NOAA')
    ax1.plot(x_plot.detach().numpy(), T_plot.detach().numpy() - 273.15, 'b-', linewidth=2.5, label='Predicción T(x)')
    ax1.plot(x_physics.detach().numpy(), [-48]*x_physics.shape[0], 'g|', markersize=8, label='Collocation Points (RAR)')
    ax1.set_xlabel(r'Variable espacial $x = \sin(latitud)$')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Perfil de Temperaturas')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    ax2.plot(x_plot.detach().numpy(), D_plot.detach().numpy(), 'm-', linewidth=2.5, label='D(x) Descubierto')
    ax2.set_xlabel(r'Variable espacial $x = \sin(latitud)$')
    ax2.set_ylabel('Difusividad (D)')
    ax2.set_title('Campo de Difusividad Térmica Oculto')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    return model, history

if __name__ == "__main__":
    trained_model, training_history = train_inverse_pinn(epochs_adam=10000, lambda_physics_max=0.5)