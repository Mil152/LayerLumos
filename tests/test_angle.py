import unittest
import numpy as np
from scipy.constants import c
from layerlumos.utils import load_material, interpolate_material
from layerlumos.layerlumos import stackrt

class TestLayerLumosStackrt(unittest.TestCase):
    def test_stackrt_with_angles(self):
        # Load material data for SiO2
        TiO2_data = load_material('TiO2')

        # Define wavelength range (in meters)
        wavelengths = np.linspace(300e-9, 900e-9, 3)  # 100 points from 300nm to 700nm
        frequencies = c / wavelengths  # Convert wavelengths to frequencies

        # Interpolate n and k values for SiO2 over the specified frequency range
        n_k_TiO2 = interpolate_material(TiO2_data, frequencies)
        n_TiO2 = n_k_TiO2[:, 0] + 1j*n_k_TiO2[:, 1]  # Combine n and k into a complex refractive index

        # Define stack configuration
        n_air = np.ones_like(wavelengths)  # Refractive index of air is approximately 1
        # Stack refractive indices and thicknesses for air-SiO2-air
        n_stack = np.vstack([n_air, n_TiO2, n_air]).T  # Transpose to match expected shape (Nlayers x Nfreq)
        d_stack = np.array([0, 2e-8, 0])  # No frequency dependence on thickness
        thetas = np.linspace(0, 89, 3)
        # Calculate R and T over the frequency (wavelength) range
        R_TE, T_TE, R_TM, T_TM = stackrt(n_stack, d_stack, frequencies, thetas)

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2

        # Expected results
        expected_R_avg = np.array([
            [0.25376038, 0.34249332, 0.98213456],
            [0.09363298, 0.16599289, 0.99560924],
            [0.04185379, 0.08031534, 0.9895475],
        ])

        expected_T_avg = np.array([
            [1.09174862e-01, 7.48343296e-02, 1.02309140e-04],
            [4.92819598e-01, 4.28186266e-01, 1.29709745e-03],
            [5.50629165e-01, 5.15418847e-01, 3.27875096e-03],
        ])

        # Calculate average R and T
        R_avg = (R_TE + R_TM) / 2
        T_avg = (T_TE + T_TM) / 2

        # Verify the results
        np.testing.assert_almost_equal(R_avg, expected_R_avg, decimal=5, err_msg="Reflectance values do not match expected results")
        np.testing.assert_almost_equal(T_avg, expected_T_avg, decimal=5, err_msg="Transmittance values do not match expected results")

if __name__ == "__main__":
    unittest.main()
