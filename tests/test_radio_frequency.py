import numpy as np

from layerlumos.utils_materials import load_material_RF
from layerlumos.layerlumos import stackrt0


def test_stackrt0_RF():
    # Define frequency range
    frequencies = np.linspace(8e9, 18e9, 100)  # Frequency range from 8GHz to 18GHz

    # Load material data for 'Ag' over the specified frequency range
    n_k_si02 = load_material_RF('Ag', frequencies)
    n_si02 = n_k_si02[:, 1] + 1j*n_k_si02[:, 2]  # Combine n and k into a complex refractive index

    # Define stack configuration
    n_air = np.ones_like(frequencies)  # Refractive index of air
    d_air = np.array([0])  # Air layer thickness
    d_si02 = np.array([5e-8])  # SiO2 layer thickness

    # Stack refractive indices and thicknesses for air-SiO2-air
    n_stack = np.vstack([n_air, n_si02, n_air]).T  # Transpose to match expected shape
    d_stack = np.vstack([d_air, d_si02, d_air])  # Stack thickness

    # Calculate R and T over the frequency range
    R_TE, T_TE, R_TM, T_TM = stackrt0(n_stack, d_stack, frequencies)

    # Calculate average shielding effectiveness
    SE_TE = -10 * np.log10(T_TE)
    SE_TM = -10 * np.log10(T_TM)
    SE = (SE_TE + SE_TM) / 2

    expected_mean_SE = 55.45905006672689
    # Assert that the actual mean SE is close to the expected value
    np.testing.assert_allclose(np.mean(SE), expected_mean_SE, rtol=1e-5)
