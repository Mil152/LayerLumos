import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c

def load_material(material_name):
    """
    Load material data from its CSV file, converting wavelength to frequency.

    Parameters:
    - material_name: The name of the material to load.
    
    Returns:
    - A NumPy array with columns for frequency (converted from wavelength), n, and k.
    """
    # Determine the directory of the current script
    current_dir = Path(__file__).parent

    # Build the absolute path to materials.json
    materials_file = current_dir / "materials.json"

    # Load the material index JSON to find the CSV file path
    with open(materials_file, 'r') as file:
        material_index = json.load(file)
    
    # Get the file path for the requested material
    relative_file_path = material_index.get(material_name)
    if not relative_file_path:
        raise ValueError(f"Material {material_name} not found in the index.")
    
    # Construct the absolute path to the material CSV file
    csv_file_path = current_dir / relative_file_path

    # Initialize a list to hold the converted data
    data = []
    
    # Open and read the CSV file
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            wavelength_um, n, k = map(float, row)
            # Convert wavelength in um to frequency in Hz
            frequency = c / (wavelength_um * 1e-6)
            data.append((frequency, n, k))
    
    # Convert to a NumPy array for easier handling in calculations
    data = np.array(data)
    # Ensure the data is sorted by frequency in ascending order
    data = data[data[:, 0].argsort()]
    
    return data

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
    
    # Remove duplicates (if any) while preserving order
    unique_freqs, indices = np.unique(freqs, return_index=True)
    unique_n_values = n_values[indices]
    unique_k_values = k_values[indices]
    
    # Ensure frequencies are sorted (usually they should be, but just in case)
    sorted_indices = np.argsort(unique_freqs)
    sorted_freqs = unique_freqs[sorted_indices]
    sorted_n_values = unique_n_values[sorted_indices]
    sorted_k_values = unique_k_values[sorted_indices]
    
    # Create interpolation functions for the sorted, unique data
    n_interp = interp1d(sorted_freqs, sorted_n_values, kind='cubic', fill_value="extrapolate")
    k_interp = interp1d(sorted_freqs, sorted_k_values, kind='cubic', fill_value="extrapolate")
    
    # Interpolate n and k for the given frequencies
    n_interp_values = n_interp(frequencies)
    k_interp_values = k_interp(frequencies)
    
    return np.vstack((n_interp_values, k_interp_values)).T