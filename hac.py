#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HAC covariance matrix estimator.

HAC - Heteroscedasticity and Autocorrelation Consistent
"""
from __future__ import division
from math import cos, sin, pi
import numpy as np

def hac(vectors, kernel='SU', band=0):
    """HAC estimator of the long-run variance matrix of u.

    Parameters
    ----------
        vectors: (T, q) array
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
    length = vectors.shape[0]

    # Demean to improve covariance estimate in small samples
    # T x q
    vectors -= vectors.mean(0)
    # q x q
    covar = np.dot(vectors.T, vectors) / length

    for lag in range(band):

        # Some constants
        a_coef = (lag+1)/(band+1)
        d_coef = (lag+1)/band
        m_coef = 6*pi*d_coef/5

        # Serially Uncorrelated
        if kernel == 'SU':
            weight = 0
        # Newey West (1987)
        elif 'Bartlett':
            if a_coef <= 1:
                weight = 1-a_coef
            else:
                weight = 0
        # Gallant (1987)
        elif kernel == 'Parzen':
            if a_coef <= .5:
                weight = 1 - 6*d_coef**2 * (1-a_coef)
            elif a_coef <= 1:
                weight = 2*(1-a_coef)**3
            else:
                weight = 0
        # Andrews (1991)
        elif kernel == 'Quadratic':
            weight = 25 / (12*(d_coef*pi)**2) \
                * (sin(m_coef)/m_coef - cos(m_coef))

        else:
            raise Exception('Kernel is not yet implemented')

        # q x q
        gamma = vectors[:-lag-1].T.dot(vectors[lag+1:]) / length
        # q x q, w is scalar
        covar += weight * (gamma + gamma.T)

    return covar
