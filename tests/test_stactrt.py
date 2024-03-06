import numpy as np
from src.layerlumos import stackrt  # Adjust the import statement based on your project structure

def test_air_glass_air_stack():
    # Define the test case for an air-glass-air stack
    n = np.array([1, 1.4, 1])  # Refractive indices for air, glass, air
    d = np.array([0, 1e-6, 0])  # Thickness in meters, with glass layer being 1 micron thick
    f = np.array([3e8 / 500e-9])  # Frequency for 500 nm wavelength
    theta = 0  # Normal incidence

    # Expected results
    # For simplicity, these are approximate values. In practice, you might calculate these
    # using known formulas for normal incidence or reference data.
    expected_R_TE = np.array([0.04])  # ~4% reflection for air-glass interface at normal incidence
    expected_T_TE = np.array([0.96])  # ~96% transmission for air-glass interface at normal incidence
    expected_R_TM = np.array([0.04])  # TM and TE are the same for normal incidence
    expected_T_TM = np.array([0.96])

    # Execute the function under test
    R_TE, T_TE, R_TM, T_TM = stackrt(n, d, f, theta)

    # Assertions to compare the actual vs. expected results
    np.testing.assert_almost_equal(R_TE, expected_R_TE, decimal=5)
    np.testing.assert_almost_equal(T_TE, expected_T_TE, decimal=5)
    np.testing.assert_almost_equal(R_TM, expected_R_TM, decimal=5)
    np.testing.assert_almost_equal(T_TM, expected_T_TM, decimal=5)
