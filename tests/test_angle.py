import unittest
import numpy as np
from scipy.constants import c
from layerlumos.utils import load_material, interpolate_material
from layerlumos.layerlumos import stackrt

class TestLayerLumosStackrt(unittest.TestCase):
    def test_stackrt_with_angles(self):
        # Load material data for SiO2
        si02_data = load_material('SiO2')

        # Define wavelength range (in meters) and convert to frequencies
        wavelengths = np.linspace(300e-9, 900e-9, 3)
        frequencies = c / wavelengths

        # Interpolate n and k values for SiO2 over the specified frequency range
        n_k_si02 = interpolate_material(si02_data, frequencies)
        n_si02 = n_k_si02[:, 0] + 1j * n_k_si02[:, 1]

        # Define stack configuration
        n_air = np.ones_like(wavelengths)
        d_air = np.array([0])
        d_si02 = np.array([2e-6])

        # Stack refractive indices and thicknesses for air-SiO2-air
        n_stack = np.vstack([n_air, n_si02, n_air]).T
        d_stack = np.vstack([d_air, d_si02, d_air])
        thetas = np.linspace(10, 40, 3)

        # Calculate R and T over the frequency (wavelength) range
        R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

        # Expected results
        expected_R_avg = np.array([
            [0.14203791, 0.00191109, 0.05613855],
            [0.12495949, 0.04856867, 0.13914601],
            [0.10354121, 0.03205252, 0.00632787]
        ])
        expected_T_avg = np.array([
            [0.85796209, 0.99808891, 0.94386145],
            [0.87504051, 0.95143133, 0.86085399],
            [0.89645879, 0.96794748, 0.99367213]
        ])

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2

        # Verify the results
        np.testing.assert_almost_equal(R_avg, expected_R_avg, decimal=5, err_msg="Reflectance values do not match expected results")
        np.testing.assert_almost_equal(T_avg, expected_T_avg, decimal=5, err_msg="Transmittance values do not match expected results")

if __name__ == "__main__":
    unittest.main()
