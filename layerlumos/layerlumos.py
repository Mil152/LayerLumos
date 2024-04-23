import numpy as np

from .utils_spectra import convert_frequencies_to_wavelengths


def stackrt_theta(n, d, f, theta=0):
    """
    Calculate the reflection and transmission coefficients for a multilayer stack
    at different frequencies under an arbitrary angle of incidence.

    :param n: The refractive indices of the layers for each frequency.
              Shape should be (Nfreq, Nlayers), where Nfreq is the number of
              frequencies and Nlayers is the number of layers.
    :param d: The thicknesses of the layers. Shape should be (Nlayers,).
    :param f: The frequencies at which to calculate the coefficients.
              Shape should be (Nfreq,).
    :param theta: The incident angle in degrees. Defaults to 0 for normal incidence.
    :returns: A tuple containing:
              - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
              - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
              - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
              - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).

    """

    assert isinstance(n, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert n.ndim == 2
    assert f.ndim == 1
    assert d.ndim == 1
    assert n.shape[0] == f.shape[0]
    assert n.shape[1] == d.shape[0]

    wvl = convert_frequencies_to_wavelengths(f)
    theta_rad = np.radians(theta)  # Convert theta to radians for calculations
    r_TE, r_TM, t_TE, t_TM = np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128)
    R_TE, T_TE, R_TM, T_TM = np.zeros_like(f), np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)

    for i, freq in enumerate(f):
        lambda_i = wvl[i]
        M_TE, M_TM = np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)
        theta_i = theta_rad

        for j in range(len(n[i, :]) - 1):
            n_j, n_next, d_next = n[i, j], n[i, j+1], d[j+1]
            # Calculate theta_t using Snell's law
            sin_theta_t = n_j * np.sin(theta_i) / n_next
            theta_t = np.arcsin(sin_theta_t)

            # Reflection and transmission coefficients for TE and TM
            # Need to be calculated using Fresnel equations including theta
            cos_theta_i, cos_theta_t = np.cos(theta_i), np.cos(theta_t)
            # For TE polarization
            r_jk_TE = (n_j * cos_theta_i - n_next * cos_theta_t) / (n_j * cos_theta_i + n_next * cos_theta_t)
            t_jk_TE = 2 * n_j * cos_theta_i / (n_j * cos_theta_i + n_next * cos_theta_t)
            # For TM polarization
            r_jk_TM = (n_next * cos_theta_i - n_j * cos_theta_t) / (n_next * cos_theta_i + n_j * cos_theta_t)
            t_jk_TM = 2 * n_j * cos_theta_i / (n_next * cos_theta_i + n_j * cos_theta_t)

            # Update matrices for TE and TM polarizations
            M_jk_TE = np.array([[1/t_jk_TE, r_jk_TE/t_jk_TE], [r_jk_TE/t_jk_TE, 1/t_jk_TE]], dtype=np.complex128)
            M_jk_TM = np.array([[1/t_jk_TM, r_jk_TM/t_jk_TM], [r_jk_TM/t_jk_TM, 1/t_jk_TM]], dtype=np.complex128)

            delta = 2 * np.pi * n_next * d_next * cos_theta_t/ lambda_i
            P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]], dtype=np.complex128)
            M_TE = np.dot(M_TE, np.dot(M_jk_TE, P))
            M_TM = np.dot(M_TM, np.dot(M_jk_TM, P))

            theta_i = theta_t  # Update theta_i for the next layer

        r_TE[i], t_TE[i] = np.nan_to_num(M_TE[1, 0] / M_TE[0, 0]), np.nan_to_num(1 / M_TE[0, 0])
        r_TM[i], t_TM[i] = np.nan_to_num(M_TM[1, 0] / M_TM[0, 0]), np.nan_to_num(1 / M_TM[0, 0])
        R_TE[i], T_TE[i] = np.abs(r_TE[i])**2, np.abs(t_TE[i])**2 * np.real(n[i, -1] * np.cos(theta_rad) / (n[i, 0] * cos_theta_t))
        R_TM[i], T_TM[i] = np.abs(r_TM[i])**2, np.abs(t_TM[i])**2 * np.real(n[i, -1] * np.cos(theta_rad) / (n[i, 0] * cos_theta_t))

    assert R_TE.ndim == 1
    assert T_TE.ndim == 1
    assert R_TM.ndim == 1
    assert T_TM.ndim == 1

    return R_TE, T_TE, R_TM, T_TM

def stackrt(n, d, f, theta=np.array([0])):
    """
    Calculate the reflection and transmission coefficients for a multilayer stack
    at different frequencies and incidence angles.

    :param n: The refractive indices of the layers for each frequency.
              Shape should be (Nfreq, Nlayers), where Nfreq is the number of
              frequencies and Nlayers is the number of layers.
    :type n: numpy.ndarray
    :param d: The thicknesses of the layers. Shape should be (Nlayers,).
    :type d: numpy.ndarray
    :param f: The frequencies at which to calculate the coefficients.
              Shape should be (Nfreq,).
    :type f: numpy.ndarray
    :param theta: The incidence angle(s) in degrees. Can be a single value or an array of angles. Default is an array containing 0.
    :type theta: float or numpy.ndarray

    :returns: A tuple containing:
              - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
              - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
              - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
              - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).

    """

    assert isinstance(n, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(theta, np.ndarray)
    assert n.ndim == 2
    assert f.ndim == 1
    assert d.ndim == 1
    assert n.shape[0] == f.shape[0]
    assert n.shape[1] == d.shape[0]

    wvl = convert_frequencies_to_wavelengths(f)
    Nfreq = len(f)
    Ntheta = len(theta)

    # Initialize arrays to hold the results
    R_TE = np.zeros((Ntheta, Nfreq))
    T_TE = np.zeros((Ntheta, Nfreq))
    R_TM = np.zeros((Ntheta, Nfreq))
    T_TM = np.zeros((Ntheta, Nfreq))

    # Iterate over each angle in theta and calculate the coefficients
    for i, angle in enumerate(theta):
        # Utilize the stackrt_theta function for each angle
        R_TE_i, T_TE_i, R_TM_i, T_TM_i = stackrt_theta(n, d, f, angle)
        R_TE[i, :], T_TE[i, :], R_TM[i, :], T_TM[i, :] = R_TE_i, T_TE_i, R_TM_i, T_TM_i

    assert R_TE.ndim == 2
    assert T_TE.ndim == 2
    assert R_TM.ndim == 2
    assert T_TM.ndim == 2

    return R_TE, T_TE, R_TM, T_TM

def stackrt0(n, d, f):
    """
    Calculate the reflection and transmission coefficients for a multilayer stack
    at different frequencies under normal incidence.

    :param n: The refractive indices of the layers for each frequency.
              Shape should be (Nfreq, Nlayers), where Nfreq is the number of
              frequencies and Nlayers is the number of layers.
    :type n: numpy.ndarray
    :param d: The thicknesses of the layers. Shape should be (Nlayers,).
    :type d: numpy.ndarray
    :param f: The frequencies at which to calculate the coefficients.
              Shape should be (Nfreq,).
    :type f: numpy.ndarray

    :returns: A tuple containing:
              - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
              - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
              - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
              - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).

    """

    wvl = convert_frequencies_to_wavelengths(f)
    # Initialize arrays for both amplitude and intensity coefficients
    r_TE, r_TM, t_TE, t_TM = np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128)
    R_TE, T_TE, R_TM, T_TM = np.zeros_like(f), np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)

    for i, freq in enumerate(f):  # Iterate over each frequency
        lambda_i = wvl[i]
        M_TE, M_TM = np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)

        for j in range(0, len(n[i, :]) - 1):
            n_j = n[i, j]
            n_next = n[i, j+1]
            d_next = d[j+1]

            # Interface calculations for TE and TM polarization
            r_jk_TE = (n_j - n_next) / (n_j + n_next)
            t_jk_TE = 2 * n_j / (n_j + n_next)
            M_jk_TE = np.array([[1/t_jk_TE, r_jk_TE/t_jk_TE], [r_jk_TE/t_jk_TE, 1/t_jk_TE]], dtype=np.complex128)
            r_jk_TM = (n_next - n_j) / (n_next + n_j)
            t_jk_TM = 2 * n_j / (n_next + n_j)
            M_jk_TM = np.array([[1/t_jk_TM, r_jk_TM/t_jk_TM], [r_jk_TM/t_jk_TM, 1/t_jk_TM]], dtype=np.complex128)

            delta = 2 * np.pi * n_next * d_next / lambda_i
            P = np.array([[np.exp(-1j * delta.item()), 0], [0, np.exp(1j * delta.item())]], dtype=np.complex128)
            M_TE = np.dot(M_TE, np.dot(M_jk_TE, P))
            M_TM = np.dot(M_TM, np.dot(M_jk_TM, P))

        r_TE[i], t_TE[i] = np.nan_to_num(M_TE[1, 0] / M_TE[0, 0]), np.nan_to_num(1 / M_TE[0, 0])
        r_TM[i], t_TM[i] = np.nan_to_num(M_TM[1, 0] / M_TM[0, 0]), np.nan_to_num(1 / M_TM[0, 0])
        R_TE[i], T_TE[i] = np.abs(r_TE[i])**2, np.abs(t_TE[i])**2 * np.real(n[i, -1] / n[i, 0])
        R_TM[i], T_TM[i] = np.abs(r_TM[i])**2, np.abs(t_TM[i])**2 * np.real(n[i, -1] / n[i, 0])

    assert R_TE.ndim == 1
    assert T_TE.ndim == 1
    assert R_TM.ndim == 1
    assert T_TM.ndim == 1

    return R_TE, T_TE, R_TM, T_TM
