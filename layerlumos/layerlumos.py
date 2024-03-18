import numpy as np
from scipy.constants import c
from .utils import load_material, interpolate_material
import numpy as np

def stackrt(n, d, f, theta=np.array([0])):
    """
    Calculates the reflection and transmission coefficients for a multilayer stack
    at different frequencies and incidence angles.

    Parameters:
    - n (numpy.ndarray): The refractive indices of the layers for each frequency.
                         Shape should be (Nfreq, Nlayers), where Nfreq is the number of
                         frequencies and Nlayers is the number of layers.
    - d (numpy.ndarray): The thicknesses of the layers. Shape should be (Nlayers,).
    - f (numpy.ndarray): The frequencies at which to calculate the coefficients.
                         Shape should be (Nfreq,).
    - theta (float or numpy.ndarray): The incidence angle(s) in degrees. Can be a single value or an array of angles.

    Returns:
    - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
    - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
    - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
    - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).
    """
    c = 3e8  # Speed of light in vacuum
    wvl = c / f  # Convert frequency to wavelength
    theta_radians = np.radians(theta)  # Convert incidence angle to radians

    # Initialize arrays for both amplitude and intensity coefficients
    r_TE, r_TM, t_TE, t_TM = (np.zeros((len(f), len(theta_radians)), dtype=np.complex128) for _ in range(4))
    R_TE, T_TE, R_TM, T_TM = (np.zeros((len(f), len(theta_radians))) for _ in range(4))

    for i, lambda_i in enumerate(wvl):
        for angle_idx, theta_i in enumerate(theta_radians):
            M_TE, M_TM = (np.eye(2, dtype=np.complex128) for _ in range(2))
            
            # Snell's law to calculate angle in each layer
            sin_theta_i = np.sin(theta_i)
            cos_theta_i = np.cos(theta_i)
            n_0 = n[i, 0]
            
            # Calculating angles for all layers
            sin_theta_layers = n_0 * sin_theta_i / n[i, :]
            cos_theta_layers = np.sqrt(1 - np.abs(sin_theta_layers)**2)  # Adjusted for complex refractive indices
            
            for j in range(len(d) - 1):
                n_j, n_next = n[i, j], n[i, j+1]
                d_j = d[j+1]
                cos_theta_j = cos_theta_layers[j]
                cos_theta_next = cos_theta_layers[j+1] if j + 1 < len(cos_theta_layers) else cos_theta_i
                
                # Checking for total internal reflection
                if np.any(np.abs(sin_theta_layers) > 1):
                    t_TE[i, angle_idx], t_TM[i, angle_idx] = 0, 0
                    R_TE[i, angle_idx], R_TM[i, angle_idx] = 1, 1
                    continue
                
                # Interface calculations for TE and TM polarization with angle consideration
                r_jk_TE = (n_j * cos_theta_j - n_next * cos_theta_next) / (n_j * cos_theta_j + n_next * cos_theta_next)
                t_jk_TE = 2 * n_j * cos_theta_j / (n_j * cos_theta_j + n_next * cos_theta_next)
                
                r_jk_TM = (n_next * cos_theta_next * n_j - n_j * cos_theta_j * n_next) / (n_next * cos_theta_next * n_j + n_j * cos_theta_j * n_next)
                t_jk_TM = 2 * n_j * cos_theta_j / (n_next * cos_theta_next * n_j + n_j * cos_theta_j * n_next)
                
                M_jk_TE = np.array([[1 / t_jk_TE, r_jk_TE / t_jk_TE], [r_jk_TE / t_jk_TE, 1 / t_jk_TE]], dtype=np.complex128)
                M_jk_TM = np.array([[1 / t_jk_TM, r_jk_TM / t_jk_TM], [r_jk_TM / t_jk_TM, 1 / t_jk_TM]], dtype=np.complex128)
                
                # Phase change calculation for angle
                delta = 2 * np.pi * n_next * d_j * cos_theta_next / lambda_i
                P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]], dtype=np.complex128)
                
                M_TE = np.dot(M_TE, np.dot(M_jk_TE, P))
                M_TM = np.dot(M_TM, np.dot(M_jk_TM, P))
            
            r_TE[i, angle_idx], t_TE[i, angle_idx] = M_TE[1, 0] / M_TE[0, 0], 1 / M_TE[0, 0]
            r_TM[i, angle_idx], t_TM[i, angle_idx] = M_TM[1, 0] / M_TM[0, 0], 1 / M_TM[0, 0]
            R_TE[i, angle_idx] = np.abs(r_TE[i, angle_idx])**2
            T_TE[i, angle_idx] = np.abs(t_TE[i, angle_idx])**2 * np.real(n[i, -1] * np.conj(cos_theta_layers[-1]) / (n_0 * cos_theta_i))
            R_TM[i, angle_idx] = np.abs(r_TM[i, angle_idx])**2
            T_TM[i, angle_idx] = np.abs(t_TM[i, angle_idx])**2 * np.real(n[i, -1] * np.conj(cos_theta_layers[-1]) / (n_0 * cos_theta_i))

    return R_TE, T_TE, R_TM, T_TM

def stackrt0(n, d, f):
    """
    Calculates the reflection and transmission coefficients for a multilayer stack
    at different frequencies under normal incidence.

    Parameters:
    - n (numpy.ndarray): The refractive indices of the layers for each frequency.
                         Shape should be (Nfreq, Nlayers), where Nfreq is the number of
                         frequencies and Nlayers is the number of layers.
    - d (numpy.ndarray): The thicknesses of the layers. Shape should be (Nlayers,).
    - f (numpy.ndarray): The frequencies at which to calculate the coefficients.
                         Shape should be (Nfreq,).

    Returns:
    - R_TE (numpy.ndarray): Reflectance for TE polarization. Shape is (Nfreq,).
    - T_TE (numpy.ndarray): Transmittance for TE polarization. Shape is (Nfreq,).
    - R_TM (numpy.ndarray): Reflectance for TM polarization. Shape is (Nfreq,).
    - T_TM (numpy.ndarray): Transmittance for TM polarization. Shape is (Nfreq,).
    """
    wvl = c / f  # Convert frequency to wavelength
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

        r_TE[i], t_TE[i] = M_TE[1, 0] / M_TE[0, 0], 1 / M_TE[0, 0]
        r_TM[i], t_TM[i] = M_TM[1, 0] / M_TM[0, 0], 1 / M_TM[0, 0]
        R_TE[i], T_TE[i] = np.abs(r_TE[i])**2, np.abs(t_TE[i])**2 * np.real(n[i, -1] / n[i, 0])
        R_TM[i], T_TM[i] = np.abs(r_TM[i])**2, np.abs(t_TM[i])**2 * np.real(n[i, -1] / n[i, 0])

    return R_TE, T_TE, R_TM, T_TM