import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Problem Parameters
D = 1.0     # Diffusion coefficient
Sigma_r = 0.1   # Removal cross-section
nu_sigma_f = 0.2  # Neutron production rate per fission
k = 1.2    # Effective multiplication factor

# Initial Conditions
L = 10.0   # Domain length
n_points = 100  # Number of grid points
x = np.linspace(0, L, n_points)  # Spatial mesh
phi_initial = np.zeros((n_points, 2))  # Initial neutron flux (2 neutron groups)
phi_initial[n_points // 2] = [1.0, 0.0]  # Initial condition: Central source for the first group

# Function to solve the multi-group diffusion equation
def diffusion_equation(phi, x):
    # Separation of flux densities for each group
    phi_group1 = phi[:n_points]
    phi_group2 = phi[n_points:]
    
    # Calculate spatial derivatives using finite differences
    dphi_dx_group1 = np.gradient(phi_group1, x)
    dphi_dx_group2 = np.gradient(phi_group2, x)
    
    # Calculate rate of change of flux densities for each group
    dphi_dt_group1 = (nu_sigma_f / k - Sigma_r) * phi_group1 - D * dphi_dx_group1
    dphi_dt_group2 = (nu_sigma_f / k - Sigma_r) * phi_group2 - D * dphi_dx_group2
    
    return np.concatenate((dphi_dt_group1, dphi_dt_group2))

# Solve the diffusion equation using integrate.odeint
sol = integrate.odeint(diffusion_equation, phi_initial.flatten(), x)

# Separate the solution for each group
sol_group1 = sol[:, :n_points]
mean = np.nanmean(sol_group1)
sol_group1[np.isnan(sol_group1)] = 0
sol_group2 = sol[:, n_points:]
mean = np.nanmean(sol_group2)
sol_group2[np.isnan(sol_group2)] = 0

# Plot the solution for each group
plt.figure(figsize=(10, 6))
plt.plot(x, sol_group1[-1], label='Group 1')
plt.plot(x, sol_group2[-1], label='Group 2')
plt.xlabel('Position (x)')
plt.ylabel('Neutron Flux Density')
plt.title('Solution of the Neutron Diffusion Equation (Simplified Example - 2 Groups)')
plt.legend()
plt.grid(True)
plt.show()

# Note: This is a simplified example considering 2 neutron groups and does not yet include temperature variations
# or specific boundary conditions of real nuclear reactors.