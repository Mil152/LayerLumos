import sys, os
sys.path.append("C:\\Program Files\\Lumerical\\v241\\api\\python\\")
import lumapi
import numpy as np

fdtd = lumapi.FDTD(hide=True)
wavelengths = np.linspace(300e-9, 900e-9, 50)  # 100 points from 300nm to 700nm
frequencies = c / wavelengths  # Convert wavelengths to frequencies

layer_materials = ['Air', 'SiO2 (Glass) - Palik', 'Air']
layer_thicknesses = [0, 2e-6, 0]
num_layers = len(layer_materials)
num_freqs = len(frequencies)
n_matrix = np.zeros((num_layers, num_freqs), dtype=np.complex128)
for i, material_name in enumerate(layer_materials):
    if material_name == 'Air':
        n_matrix[i, :] = 1
    else:
        n_complex = fdtd_solver.get_index(str(material_name), frequencies)
        n_matrix[i, :] = n_complex[:, 0]
theta = np.linspace(0, 89, 90)
RT = fdtd.stackrt(n_matrix, layer_thicknesses, frequencies, theta)