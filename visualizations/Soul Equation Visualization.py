import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
time_steps = 100
known_growth_rate = 1.05
unknown_decay_rate = 0.95

# Initialize variables
M_values = [1]  # Start with initial knowns
neg_M_values = [-1]  # Start with initial unknowns
consciousness_values = []  # Imaginary component (i)

# Simulate dynamics
for t in range(time_steps):
    M = M_values[-1] * known_growth_rate
    neg_M = neg_M_values[-1] * unknown_decay_rate
    consciousness = np.exp(1j * np.pi * (t + 1) / time_steps)  # Simulate consciousness as a phase shift

    # Append to lists
    M_values.append(M)
    neg_M_values.append(neg_M)
    consciousness_values.append(consciousness)

# Prepare arrays for Universe (U) computation
M_array = np.array(M_values[:-1])  # Slice to match length of consciousness_array
neg_M_array = np.array(neg_M_values[:-1])  # Slice to match length of consciousness_array
consciousness_array = np.array(consciousness_values)

# Compute Universe (U)
U_array = (M_array + neg_M_array) ** consciousness_array

# Real and imaginary parts of U
U_real = U_array.real
U_imag = U_array.imag

# Create a meshgrid for visualization
T = np.arange(1, time_steps + 1)
M_mesh, T_mesh = np.meshgrid(M_array, T)

# Broadcast real and imaginary parts for 2D visualization
U_real_2d = np.broadcast_to(U_real[:, np.newaxis], T_mesh.shape)
U_imag_2d = np.broadcast_to(U_imag[:, np.newaxis], T_mesh.shape)

# Visualization
fig = plt.figure(figsize=(12, 6))

# Real part visualization
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T_mesh, M_mesh, U_real_2d, cmap='viridis', edgecolor='none')
ax1.set_title('Real Part of U')
ax1.set_xlabel('Time')
ax1.set_ylabel('M')
ax1.set_zlabel('Real(U)')
ax1.view_init(elev=30, azim=90)  # Adjust view for rotation

# Imaginary part visualization
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T_mesh, M_mesh, U_imag_2d, cmap='plasma', edgecolor='none')
ax2.set_title('Imaginary Part of U')
ax2.set_xlabel('Time')
ax2.set_ylabel('M')
ax2.set_zlabel('Imag(U)')
ax2.view_init(elev=30, azim=90)  # Adjust view for rotation

plt.tight_layout()
plt.show()
