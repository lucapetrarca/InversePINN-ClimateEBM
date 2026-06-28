import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.model import ClimateInversePINN
from src.physics import calculate_physics_loss
from src.data_processing import get_training_data, get_collocation_points, get_rar_collocation_points

def train_inverse_pinn(epochs_adam=5000, lambda_physics_max=0.5):
    print("Iniciando entrenamiento de la PINN")
    
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
               'D': [], 'A': [], 'B': [], 'C': []}
    
    print("\nFASE 1: Adam con RAR")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        
        lambda_val = lambda_physics_max * (epoch / epochs_adam)
        
        T_pred = model(x_data)
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
            D_val = F.softplus(model.D).item()
            A_val = model.A_out.item()
            B_val = model.B_out.item()
            C_val = model.C_out.item()
            
            history['loss_total'].append(loss_total.item())
            history['D'].append(D_val)
            history['loss_data'].append(loss_data.item())
            history['loss_physics'].append(loss_physics.item())
            history['A'].append(A_val)
            history['B'].append(B_val)
            history['C'].append(C_val)
            
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch:04d} | L_Total: {loss_total.item():.2f} | L_Data: {loss_data.item():.2f} | L_Phys: {loss_physics.item():.2f}")

    print("\n--- FASE 2: Entrenando con L-BFGS (Ajuste Fino Profundo) ---")
    
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
        
        T_pred = model(x_data)
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
    
    T_pred_final = model(x_data)
    final_L_data = torch.mean((T_pred_final - T_data)**2).item()
    final_L_phys_colloc = calculate_physics_loss(model, x_physics).item()
    final_L_phys_data = calculate_physics_loss(model, x_data).item()
    
    print("\n--- VALIDACIÓN CIENTÍFICA ---")
    print(f"1. Error Estadístico (L_Data MSE):        {final_L_data:.4f} °C^2")
    print(f"2. Error Físico en Sensores (L_Phys):     {final_L_phys_data:.4f}")
    print(f"3. Error de Generalización (L_Colloc):    {final_L_phys_colloc:.4f} <-- ESTA ES LA VALIDACIÓN")
    print(f"   (Física cumplida en {x_physics.shape[0]} zonas no observadas)")
    
    final_D = F.softplus(model.D).item()
    print(f"\nParámetros Descubiertos (Físicamente Válidos):")
    print(f"D={final_D:.4f}, A={model.A_out.item():.2f}, B={model.B_out.item():.4f}, C={model.C_out.item():.4f}")
    
    print("\nGenerando gráfico de validación...")
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 200).view(-1, 1)
        T_plot = model(x_plot)
        
    plt.figure(figsize=(9, 6))
    plt.plot(x_data.detach().numpy(), T_data.detach().numpy() - 273.15, 'ro', label='Datos Reales NOAA (CSV)')
    plt.plot(x_plot.detach().numpy(), T_plot.detach().numpy() - 273.15, 'b-', linewidth=2.5, label='Curva Continua Inverse PINN')
    
    plt.plot(x_physics.detach().numpy(), [-48]*x_physics.shape[0], 'g|', markersize=8, label='Collocation Points (RAR)')

    plt.xlabel(r'Variable espacial $x = \sin(latitud)$')
    plt.ylabel('Temperatura (°C)')
    plt.title('Validación del EBM Topológico (Física Restringida)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return model, history

if __name__ == "__main__":
    trained_model, training_history = train_inverse_pinn(epochs_adam=10000, lambda_physics_max=0.5)