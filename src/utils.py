import csv
import json
import numpy as np
from scipy.interpolate import interp1d

def load_material(material_name):
    """
    Load material data from its CSV file, converting wavelength to frequency.

    Parameters:
    - material_name: The name of the material to load.
    
    Returns:
    - A NumPy array with columns for frequency (converted from wavelength), n, and k.
    """
    # Load the material index JSON to find the CSV file path
    with open('src/materials.json', 'r') as file:
        material_index = json.load(file)
    
    # Get the file path for the requested material
    file_path = material_index.get(material_name)
    if not file_path:
        raise ValueError(f"Material {material_name} not found in the index.")
    
    # Speed of light in vacuum (m/s)
    c = 3e8
    # Initialize a list to hold the converted data
    data = []
    
    # Open and read the CSV file
    with open(f'src/{file_path}', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            wavelength_um, n, k = map(float, row)
            # Convert wavelength in um to frequency in Hz
            frequency = c / (wavelength_um * 1e-6)
            data.append((frequency, n, k))
    
    # Convert to a NumPy array for easier handling in calculations
    return np.array(data)

def interpolate_material(material_data, frequencies):
    """
    Interpolate n and k values for the specified frequencies.

    Parameters:
    - material_data: The data for the material, as returned by load_material.
    - frequencies: A list or NumPy array of frequencies to interpolate n and k for.
    
    Returns:
    - Interpolated values of n and k as a NumPy array.
    """
    # Extract frequency, n, and k columns
    freqs, n_values, k_values = material_data.T
    
    # Create interpolation functions
    n_interp = interp1d(freqs, n_values, kind='cubic', fill_value="extrapolate")
    k_interp = interp1d(freqs, k_values, kind='cubic', fill_value="extrapolate")
    
    # Interpolate n and k for the given frequencies
    n_interp_values = n_interp(frequencies)
    k_interp_values = k_interp(frequencies)
    
    return np.vstack((n_interp_values, k_interp_values)).T