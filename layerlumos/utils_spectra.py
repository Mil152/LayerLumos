import numpy as np
import scipy.constants as scic


def convert_frequencies_to_wavelengths(f):
    """
    Convert frequency to wavelength.

    """

    assert isinstance(f, (np.ndarray, float, np.float32, np.float64))
    if isinstance(f, np.ndarray):
        assert f.ndim == 1

    wvl = scic.c / f
    return wvl
