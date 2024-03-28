import unittest
import numpy as np
import scipy.constants as scic

from layerlumos.utils_materials import load_material, interpolate_material
from layerlumos.layerlumos import stackrt0


class TestLayerLumos(unittest.TestCase):
    def test_stackrt0(self):
        # Load material data for SiO2 (example uses 'Ag', replace with 'SiO2' or appropriate material)
        si02_data = load_material('Ag')

        # Define a small wavelength range for testing
        wavelengths = np.linspace(300e-9, 900e-9, 3)  # from 300nm to 700nm
        frequencies = scic.c / wavelengths  # Convert wavelengths to frequencies

        # Interpolate n and k values for SiO2 over the specified frequency range
        n_k_si02 = interpolate_material(si02_data, frequencies)
        n_si02 = n_k_si02[:, 0] + 1j * n_k_si02[:, 1]

        # Stack configuration
        n_air = np.ones_like(wavelengths)
        d_air = np.array([0])
        d_si02 = np.array([2e-6])

        n_stack = np.vstack([n_air, n_si02, n_air]).T
        d_stack = np.vstack([d_air, d_si02, d_air])

        # Calculate R and T over the frequency range
        R_TE, T_TE, R_TM, T_TM = stackrt0(n_stack, d_stack, frequencies)

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2

        # Expected output (based on your script; might need adjustment based on actual results)
        expected_R_avg = np.array([0.15756072, 0.98613162, 0.99031732])
        expected_T_avg = np.array([5.87146690e-34, 1.58748575e-71, 5.13012036e-76])

        # Validate the results with a tolerance for floating-point arithmetic
        np.testing.assert_allclose(R_avg, expected_R_avg, rtol=1e-2, atol=0)
        np.testing.assert_allclose(T_avg, expected_T_avg, rtol=1e-2, atol=0)


if __name__ == '__main__':
    unittest.main()
