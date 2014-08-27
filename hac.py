import numpy as np

def hac(u, kernel = 'Bartlett', band = 0):
    """HAC estimator of the long-run variance matrix of moments.
    
    Args:
        u: 2d-array, T x q
        kernel: type of kernel.
            Currenly implemented: SU, Bartlett, Parzen, Quadratic
        band: truncation parameter.
            Ideally should be chosen optimally depending on the sample size!
        
    Returns:
        S: long-run variance matrix of moments, q x q, moments x moments
        
    """
    T = u.shape[0]
    
    # Demean to improve covariance estimate in small samples
    # T x q
    u = u - u.mean(0)
    # q x q
    S = np.dot(u.T, u) / T
    
    for lag in range(band):
        
        # Some constants
        a = (lag + 1) / (band + 1)
        d = (lag + 1) / band
        m = 6 * np.pi * d / 5
        
        # Serially Uncorrelated
        if kernel == 'SU' :
            w = 0
        # Newey West (1987)
        elif 'Bartlett' :
            if a <= 1:
                w = 1 - a
            else:
                w = 0
        # Gallant (1987)
        elif kernel == 'Parzen' :
            if a <= .5:
                w = 1. - 6. * a**2 * (1. - a)
            elif a <= 1.:
                w = 2 * (1. - a)**3
            else:
                w = 0.
        # Andrews (1991)
        elif kernel == 'Quadratic' :
            w = 25 / (12*(d*np.pi)**2) * (np.sin(m)/m - np.cos(m))
        
        # q x q
        Gamma = np.dot(u[:-lag-1, :].T, u[lag+1:, :]) / T
        # q x q, w is scalar
        S += w * (Gamma + Gamma.T)
    
    return S