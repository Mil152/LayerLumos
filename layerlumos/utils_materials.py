import numpy as np
import csv
import json
from pathlib import Path
import scipy.interpolate as scii
import scipy.constants as scic

from .utils_spectra import convert_frequencies_to_wavelengths


Metals_sigma = {
    'Cu': 5.96e7,
    'Cr': 7.74e6,
    'Ag': 6.3e7,
    'Al': 3.77e7,
    'Ni' : 1.43e7,
    'W' : 1.79e7,
    'Ti' : 2.38e6,
    'Pd' : 9.52e6
}

Metals_nk_updated_specific_sigma = {}
mu0 = 4 * np.pi * 1e-7  # H/m
Z_0 = scic.physical_constants['characteristic impedance of vacuum'][0]  # Ohms, impedance of free space
# Frequency range
nu = np.linspace(8e9, 18e9, 11)  # From 8 GHz to 18 GHz
omega = 2 * np.pi * nu  # Angular frequency

for metal, sigma in Metals_sigma.items():
    Z = np.sqrt(1j * omega * mu0 / sigma)  # Impedance of the material using specific sigma
    n_complex = Z_0 / Z  # Complex refractive index

    # Extract real and imaginary parts of the refractive index
    n_real = np.real(n_complex)
    k_imag = np.imag(n_complex)

    # Update the dictionary with the new values
    Metals_nk_updated_specific_sigma[metal] = {
        'freq_data': nu.tolist(),
        'n_data': n_real.tolist(),
        'k_data': k_imag.tolist()
    }


def load_material_RF(material_name, frequencies):
    """
    Load material RF data for a given material and frequencies.

    Parameters:
    - material_name: The name of the material to load.
    - frequencies: Array of frequencies for which data is requested.

    Returns:
    - A NumPy array with columns for frequency, n, and k.

    """

    # Check if the material is defined
    if material_name not in Metals_nk_updated_specific_sigma:
        # Material not found, return default values (n=1, k=0) for given frequencies
        n_default = np.ones_like(frequencies)
        k_default = np.zeros_like(frequencies)
        data = np.column_stack((frequencies, n_default, k_default))
    else:
        # Material found, extract the data
        material_data = Metals_nk_updated_specific_sigma[material_name]
        freq_data = np.array(material_data['freq_data'])
        n_data = np.array(material_data['n_data'])
        k_data = np.array(material_data['k_data'])

        # Interpolate n and k data for the input frequencies, if necessary
        n_interpolated = np.interp(frequencies, freq_data, n_data)
        k_interpolated = np.interp(frequencies, freq_data, k_data)

        # Combine the data
        data = np.column_stack((frequencies, n_interpolated, k_interpolated))

    return data

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
            try:
                # Attempt to convert wavelength, n, and k to floats
                wavelength_um, n, k = map(float, row)
                # Convert wavelength in um to frequency in Hz
                frequency = convert_frequencies_to_wavelengths(wavelength_um * 1e-6)
                data.append((frequency, n, k))
            except ValueError:
                # If conversion fails, skip this row
                continue

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
    n_interp = scii.interp1d(sorted_freqs, sorted_n_values, kind='cubic', fill_value="extrapolate")
    k_interp = scii.interp1d(sorted_freqs, sorted_k_values, kind='cubic', fill_value="extrapolate")

    # Interpolate n and k for the given frequencies
    n_interp_values = n_interp(frequencies)
    k_interp_values = k_interp(frequencies)

    return np.vstack((n_interp_values, k_interp_values)).T
