import sys, os
sys.path.append("C:\\Program Files\\Lumerical\\v241\\api\\python\\")
sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\")
import lumapi
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

fdtd = lumapi.FDTD(hide=True)
wavelengths = np.linspace(300e-9, 900e-9, 50)  # 100 points from 300nm to 700nm
frequencies = c / wavelengths  # Convert wavelengths to frequencies

layer_materials = ['Air', 'TiO2 (Titanium Dioxide) - Siefke', 'Air']
layer_thicknesses = np.array([0, 20e-9, 0])
num_layers = len(layer_materials)
num_freqs = len(frequencies)
n_matrix = np.zeros((num_layers, num_freqs), dtype=np.complex128)
for i, material_name in enumerate(layer_materials):
    if material_name == 'Air':
        n_matrix[i, :] = 1
    else:
        n_complex = fdtd.getindex(str(material_name), frequencies)
        n_matrix[i, :] = n_complex[:, 0]
theta = np.linspace(0, 89, 90)
RT_lumerical = fdtd.stackrt(n_matrix, layer_thicknesses, frequencies, theta)

R_Lumercal = (RT_lumerical['Rp'] + RT_lumerical['Rs']) / 2
T_Lumercal = (RT_lumerical['Tp'] + RT_lumerical['Ts']) / 2

wavelengths_nm = wavelengths * 1e9

# Assuming 'theta' represents incidence angles in degrees

# Create the plots
fig, ax = plt.subplots(figsize=(10, 6))

# Reflectance heatmap
# The 'extent' argument defines the data bounds for the axes: [x_min, x_max, y_min, y_max]
refl = ax.imshow(T_Lumercal.T, extent=(wavelengths_nm.min(), wavelengths_nm.max(), theta.min(), theta.max()), 
                 origin='lower', aspect='auto', cmap='viridis')
ax.set_title('Transmittance')
ax.set_ylabel('Incidence Angle (degrees)')
ax.set_xlabel('Wavelength (nm)')
fig.colorbar(refl, ax=ax, label='Transmittance')

plt.tight_layout()
plt.show()