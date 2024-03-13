import numpy as np
from scipy.constants import c
from layerlumos.utils import load_material, interpolate_material, load_material_RF
from layerlumos.layerlumos import stackrt, stackrt0
import matplotlib.pyplot as plt
import numpy as np

# si02_data = load_material('Ag')

# Define wavelength range (in meters)
# wavelengths = np.linspace(300e-9, 900e-9, 3)  # 100 points from 300nm to 700nm
frequencies = np.linspace(8e9, 18e9, 100)  # Convert wavelengths to frequencies

# Interpolate n and k values for SiO2 over the specified frequency range
n_k_si02 = load_material_RF('Ag', frequencies)
n_si02 = n_k_si02[:, 1] + 1j*n_k_si02[:, 2]  # Combine n and k into a complex refractive index

print(n_si02)

# Define stack configuration
n_air = np.ones_like(frequencies)  # Refractive index of air is approximately 1
d_air = np.array([0])
d_si02 = np.array([5e-8])  # Thickness of SiO2 layer in meters (e.g., 2 microns)

# Stack refractive indices and thicknesses for air-SiO2-air
n_stack = np.vstack([n_air, n_si02, n_air]).T  # Transpose to match expected shape (Nlayers x Nfreq)
d_stack = np.vstack([d_air, d_si02, d_air])  # No frequency dependence on thickness

# Calculate R and T over the frequency (wavelength) range
R_TE, T_TE, R_TM, T_TM = stackrt0(n_stack, d_stack, frequencies)

# Calculate average R and T
SE_TE = -10 * np.log10(T_TE)
SE_TM = -10 * np.log10(T_TM)
SE = (SE_TE + SE_TM) / 2

# print(SE)
