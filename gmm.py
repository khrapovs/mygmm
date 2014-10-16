#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GMM estimator.

"""
from __future__ import print_function, division

import numpy as np
from scipy import stats, linalg
from scipy.optimize import minimize

from MyGMM.hac import hac

class GMM(object):
    """GMM estimation class.

    """

    def __init__(self):

        # Default options:
        self.options = {'iter' : 2}
        # Maximum iterations for the optimizer
        self.maxiter = None
        # Optimization method
        self.method = 'BFGS'
        # Display convergence results
        self.disp = False
        # Use analytic Jacobian?
        self.use_jacob = True
        # HAC kernel type
        self.kernel = 'Bartlett'
        # J-statistic
        self.jstat = None
        # Optimization results
        self.res = None
        # Standard errors
        self.se = None
        # T-statistics
        self.tstat = None
        # P-values
        self.pval = None

    def moment(self, theta):
        """Moment function.

        Computes moment restrictions and their gradients.
        Should be written for each specific problem.

        Parameters
        ----------
            theta: (k,) array
                Parameters

        Returns
        -------
            g : (T, q) array
                Matrix of moment restrictions
            dg : (q, k)
                Gradient of moment restrictions. Mean over observations

        """
        pass

    def print_results(self):
        """Print GMM estimation results.

        """
        np.set_printoptions(precision=3, suppress=True)

        print('-' * 60)
        print('The final results are')
        print(self.res.message)
        print('theta   = ', self.theta)
        print('s.e.    = ', self.se)
        print('t-stat  = ', self.tstat)
        print('J-stat  = %0.2f' % self.jstat)
        print('df      = ', self.df)
        print('p-value = %0.2f' % self.pval)
        print('-' * 60)

    def gmmest(self, theta_start, **kwargs):
        """Multiple step GMM estimation procedure.

        """
        self.options.update(kwargs)
        
        self.theta = theta_start.copy()
        g = self.moment(self.theta)[0]
        T, q = g.shape
        k = len(self.theta)
        # Number of degrees of freedom
        self.df = q - k
        

        # Weighting matrix
        self.W = np.eye(q)
        # First step GMM
        for i in range(self.options['iter']):
            # Compute optimal weighting matrix
            # Only after the first step
            if i > 0:
                g = self.moment(self.theta)[0]
                self.W = self.weights(g)

            opt_options = {'disp' : self.disp, 'maxiter' : self.maxiter}
            self.res = minimize(self.gmmobjective, self.theta,
                                method=self.method,
                                jac=self.use_jacob,
                                options=opt_options,
                                callback=self.callback)
            # Update parameter for the next step
            self.theta = self.res.x

        # k x k
        V = self.varest(self.theta)
        # J-statistic
        self.jstat = self.res.fun * T
        # p-value of the J-test, scalar
        self.pval = 1 - stats.chi2.cdf(self.jstat, self.df)
        # t-stat for each parameter, 1 x k
        self.se = np.diag(V)**.5
        # t-stat for each parameter, 1 x k
        self.tstat = self.theta / self.se

    def callback(self, theta):
        """Callback function. Prints at each optimization iteration."""
        pass

    def gmmobjective(self, theta):
        """GMM objective function and its gradient.

        Parameters
        ----------
            theta: (k,) array
                Parameters

        Returns
        -------
            f : float
                Value of objective function, see Hansen (2012, p.241)
            df : (k,) array
                Derivative of objective function.
                Depends on the switch 'use_jacob'
        """
        #theta = theta.flatten()
        # g - T x q, time x number of moments
        # dg - q x k, time x number of moments
        g, dg = self.moment(theta)
        # g - 1 x q, 1 x number of moments
        g = g.mean(0).flatten()
        # 1 x 1
        f = float(g.dot(self.W).dot(g.T))
        assert f >= 0, 'Objective function should be non-negative'

        if self.use_jacob:
            # 1 x k
            df = 2 * g.dot(self.W).dot(dg).flatten()

            return f, df
        else:
            return f

    def weights(self, g):
        """
        Optimal weighting matrix

        Parameters
        ----------
            g : (T, q) array
                Moment restrictions

        Returns
        -------
            invS : (q, q) array
                Inverse of moments covariance matrix

        """
        # q x q
        S = hac(g, **self.options)

        return linalg.pinv(S)

    def varest(self, theta):
        """Estimate variance matrix of parameters.

        Parameters
        ----------
            theta : (k,)
                Parameters

        Returns
        -------
            V : (k, k) array
                Variance matrix of parameters

        """
        # g - T x q, time x number of moments
        # dg - q x k, time x number of moments
        g, dg = self.moment(theta)
        # q x q
        S = self.weights(g)
        # k x k
        # What if k = 1?
        V = linalg.pinv(dg.T.dot(S).dot(dg)) / g.shape[0]

        return V

if __name__ == '__main__':
    import test_mygmm
    test_mygmm.test_mygmm()
