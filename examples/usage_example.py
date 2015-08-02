#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Examples of using GMM class.

"""
import numpy as np
import pandas as pd

from mygmm import GMM


class Model(object):

    """Model moment restrictions and Jacobian.

    """

    def __init__(self, data):
        self.data = data

    def momcond(self, theta, **kwargs):
        """Moment function.

        Computes momcond restrictions and their gradients.
        Should be written for each specific problem.

        Parameters
        ----------
        theta : (k,) array
            Parameters
        kwargs : dict
            Any additional keyword arguments

        Returns
        -------
        moment : (T, q) array
            Matrix of momcond restrictions
        dmoment : (q, k) array
            Gradient of momcond restrictions. Mean over observations

        """
        # T x 1
        error = self.data['Y'] - self.data['X'].dot(theta)
        # T x k
        de = -self.data['X']
        # T x q
        g = (error * self.data['Z'].T).T
        # q x k
        dg = (de[:, np.newaxis, :] * self.data['Z'][:, :, np.newaxis]).mean(0)

        return g, None

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start, **kwargs)


def simulate_data():
    """Simulate data.

    Returns
    -------
    data : dict
        Dictionary of model specific data arrays
    beta : array
        True parameter

    """
    # Number of observations
    T = 1e3
    # Correlation
    rho = .9
    # True parameter
    beta = np.array([1., -.5])
    # True parameters for instruments
    gamma = np.array([1, -5, 2, 3, -1])
    # Random errors
    e = np.random.normal(size=(T, 2))
    # Instruments
    Z = np.random.normal(size=(T, 5))
    # Endogenous variables
    X1 = Z.dot(gamma) + e[:,0]
    X2 = (Z**2).dot(gamma) + e[:,1]
    X = np.concatenate((X1[:, np.newaxis], X2[:, np.newaxis]), axis=1)
    # Dependent variable
    Y = X.dot(beta) + e[:,0] + rho * e[:,1]

    #plt.scatter(X, Y)
    #plt.show()

    # Collect data for GMM
    data = {'Y': Y, 'X': X, 'Z': Z}

    return data, beta


def try_mygmm():

    options = {'iter': 2, 'bounds': None,
               'use_jacob': True, 'method': 'L-BFGS-B'}
    data, theta_true = simulate_data()
    # Initialize GMM object
    model = Model(data)
    # Estimate model with GMM
    res_gmm = model.gmmest(theta_true*2, **options)
    # Print results
    print(res_gmm)

    # Compare with OLS
    df = pd.DataFrame(data['X'], columns=['X1', 'X2'])
    df['Y'] = data['Y']
    res = pd.ols(y=df['Y'], x=df[['X1', 'X2']], intercept=False)

    print('\nOLS results')
    print(np.array(res.beta))
    print(np.array(res.t_stat))

    return res_gmm


if __name__ == '__main__':

    res = try_mygmm()
