import timeit

# Define the setup code (import statements, variable definitions, etc.)
setup_code1 = '''
import numpy as np
from scipy.constants import c
from layerlumos.utils import load_material, interpolate_material
from layerlumos.layerlumos import stackrt, stackrt0
import numpy as np
'''

setup_code2 = '''
import sys, os
import lumapi
import numpy as np
from scipy.constants import c
'''

# Define the code snippets to be tested
code_snippet1 = '''
si02_data = load_material('SiO2')

# Define wavelength range (in meters)
wavelengths = np.linspace(300e-9, 900e-9, 50)  # 100 points from 300nm to 700nm
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
thetas = np.linspace(0, 89, 90)
# Calculate R and T over the frequency (wavelength) range
R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

'''

code_snippet2 = '''

fdtd = lumapi.FDTD(hide=True)
wavelengths = np.linspace(300e-9, 900e-9, 50)  # 100 points from 300nm to 700nm
frequencies = c / wavelengths  # Convert wavelengths to frequencies

layer_materials = ['Air', 'SiO2 (Glass) - Palik', 'Air']
layer_thicknesses = np.array([0, 2e-6, 0])
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
RT = fdtd.stackrt(n_matrix, layer_thicknesses, frequencies, theta)
'''

# Run timeit
time1 = timeit.timeit(stmt=code_snippet1, setup=setup_code1, number=10)
time2 = timeit.timeit(stmt=code_snippet2, setup=setup_code2, number=10)

print(f"Code snippet 1 execution time: {time1}")
print(f"Code snippet 2 execution time: {time2}")
