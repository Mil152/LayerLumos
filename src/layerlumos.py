import numpy as np
from scipy.constants import c
from .utils import load_material, interpolate_material

import numpy as np

def stackrt(n, d, f, theta=0):

    wvl = c / f  # Convert frequency to wavelength
    theta = np.radians(theta)  # Convert incidence angle to radians
    
    # Ensure n is a numpy array to facilitate complex number operations
    n = np.array(n, dtype=np.complex128)

    # Calculate sine and cosine of theta in each layer, considering Snell's law
    sin_theta0 = np.sin(theta)
    cos_theta = np.sqrt(1 - (n[0] * sin_theta0 / n)**2)
    
    # Initialize arrays for reflection and transmission coefficients
    R_TE, T_TE = np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128)
    R_TM, T_TM = np.zeros_like(f, dtype=np.complex128), np.zeros_like(f, dtype=np.complex128)

    for i, lambda_i in enumerate(wvl):
        M_TE, M_TM = np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)

        for j in range(1, len(n) - 1):
            # Interface matrices for TE and TM
            r_jk_TE = (n[j-1] * cos_theta[j-1] - n[j] * cos_theta[j]) / (n[j-1] * cos_theta[j-1] + n[j] * cos_theta[j])
            t_jk_TE = (2 * n[j-1] * cos_theta[j-1]) / (n[j-1] * cos_theta[j-1] + n[j] * cos_theta[j])
            r_jk_TM = (n[j] * cos_theta[j-1] - n[j-1] * cos_theta[j]) / (n[j] * cos_theta[j-1] + n[j-1] * cos_theta[j])
            t_jk_TM = (2 * n[j-1] * cos_theta[j-1]) / (n[j] * cos_theta[j-1] + n[j-1] * cos_theta[j])
            
            M_jk_TE = np.array([[1/t_jk_TE, r_jk_TE/t_jk_TE], [r_jk_TE/t_jk_TE, 1/t_jk_TE]], dtype=np.complex128)
            M_jk_TM = np.array([[1/t_jk_TM, r_jk_TM/t_jk_TM], [r_jk_TM/t_jk_TM, 1/t_jk_TM]], dtype=np.complex128)
            
            # Phase change matrix
            delta = 2 * np.pi * n[j] * d[j] * cos_theta[j] / lambda_i
            P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]], dtype=np.complex128)
            
            # Update total matrices
            M_TE = np.dot(M_TE, np.dot(M_jk_TE, P))
            M_TM = np.dot(M_TM, np.dot(M_jk_TM, P))
        
        # Calculate and store R and T for TE and TM
        r_TE, t_TE = M_TE[1, 0] / M_TE[0, 0], 1 / M_TE[0, 0]
        r_TM, t_TM = M_TM[1, 0] / M_TM[0, 0], 1 / M_TM[0, 0]
        R_TE[i], T_TE[i] = np.abs(r_TE)**2, np.abs(t_TE)**2 * (np.real(n[-1] / n[0]) * np.real(cos_theta[-1] / cos_theta[0]))
        R_TM[i], T_TM[i] = np.abs(r_TM)**2, np.abs(t_TM)**2 * (np.real(n[-1] / n[0]) * np.real(cos_theta[-1] / cos_theta[0]))

    return r_TE, t_TE, r_TM, t_TM

