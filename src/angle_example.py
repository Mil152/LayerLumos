import numpy as np
from scipy.constants import c
from utils import load_material, interpolate_material
from layerlumos import stackrt, stackrt0
import matplotlib.pyplot as plt
import numpy as np

si02_data = load_material('SiO2')

# Define wavelength range (in meters)
wavelengths = np.linspace(300e-9, 900e-9, 3)  # 100 points from 300nm to 700nm
frequencies = c / wavelengths  # Convert wavelengths to frequencies

# Interpolate n and k values for SiO2 over the specified frequency range
n_k_si02 = interpolate_material(si02_data, frequencies)
n_si02 = n_k_si02[:, 0] + 1j*n_k_si02[:, 1]  # Combine n and k into a complex refractive index

# Define stack configuration
n_air = np.ones_like(wavelengths)  # Refractive index of air is approximately 1
d_air = np.array([0])
d_si02 = np.array([2e-6])  # Thickness of SiO2 layer in meters (e.g., 2 microns)

# Stack refractive indices and thicknesses for air-SiO2-air
n_stack = np.vstack([n_air, n_si02, n_air]).T  # Transpose to match expected shape (Nlayers x Nfreq)
d_stack = np.vstack([d_air, d_si02, d_air])  # No frequency dependence on thickness
thetas = np.linspace(10, 40, 3)
# Calculate R and T over the frequency (wavelength) range
R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

# Calculate average R and T
R_avg = (R_TE + R_TM) / 2
T_avg = (T_TE + T_TM) / 2
print(R_avg)
print(T_avg)

# print(f'Reflectance range: {R_avg.min()} to {R_avg.max()}')
# print(f'Transmittance range: {T_avg.min()} to {T_avg.max()}')

# Assuming wavelengths and thetas are defined as in your example
wavelength_range = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths) + 1)
theta_range = np.linspace(min(thetas), max(thetas), len(thetas) + 1)

wavelengths_nm = wavelengths * 1e9  # Convert to nm for easier interpretation
theta_degrees = thetas  # Assuming 'thetas' is already in degrees

# Generate the grid for plotting. Note: Add one to thetas and wavelengths dimensions if using 'shading=flat'
W, T = np.meshgrid(wavelengths_nm, theta_degrees)

# Create the plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Reflectance
refl = ax[0].imshow(R_avg.T, extent=(wavelengths_nm.min(), wavelengths_nm.max(), theta_degrees.min(), theta_degrees.max()), origin='lower', aspect='auto', cmap='viridis')
ax[0].set_title('Reflectance')
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel('Incidence Angle (degrees)')
fig.colorbar(refl, ax=ax[0])

# Transmittance
tran = ax[1].imshow(T_avg.T, extent=(wavelengths_nm.min(), wavelengths_nm.max(), theta_degrees.min(), theta_degrees.max()), origin='lower', aspect='auto', cmap='viridis')
ax[1].set_title('Transmittance')
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel('Incidence Angle (degrees)')
fig.colorbar(tran, ax=ax[1])

plt.tight_layout()
plt.show()