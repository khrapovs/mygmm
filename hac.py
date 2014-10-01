#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HAC covariance matrix estimator.

HAC - Heteroscedasticity and Autocorrelation Consistent 
"""
from __future__ import division
from math import cos, sin, pi
import numpy as np

def hac(u, kernel='SU', band=0):
    """HAC estimator of the long-run variance matrix of u.
    
    Parameters
    ----------
        u: (T, q) array
            The set of q vectors for estimation of their covariance matrix.
        kernel: str
            Type of kernel.
            Currenly implemented: SU, Bartlett, Parzen, Quadratic
        band: int
            Truncation parameter.
            Ideally should be chosen optimally depending on the sample size!
        
    Returns
    -------
        S: (q, q) array
            Long-run variance matrix of u
        
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
        m = 6 * pi * d / 5
        
        # Serially Uncorrelated
        if kernel == 'SU':
            w = 0
        # Newey West (1987)
        elif 'Bartlett':
            if a <= 1:
                w = 1 - a
            else:
                w = 0
        # Gallant (1987)
        elif kernel == 'Parzen':
            if a <= .5:
                w = 1 - 6 * a**2 * (1 - a)
            elif a <= 1.:
                w = 2 * (1 - a)**3
            else:
                w = 0
        # Andrews (1991)
        elif kernel == 'Quadratic':
            w = 25 / (12*(d*pi)**2) * (sin(m)/m - cos(m))
        
        else:
            raise Exception('Kernel is not yet implemented')
        
        # q x q
        Gamma = np.dot(u[:-lag-1, :].T, u[lag+1:, :]) / T
        # q x q, w is scalar
        S += w * (Gamma + Gamma.T)
    
    return S