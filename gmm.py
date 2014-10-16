#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GMM estimator.

"""
from __future__ import print_function, division

import numpy as np
from scipy.linalg import pinv
from scipy.stats import chi2
from scipy.optimize import minimize

from MyGMM.hac import hac

class Results(object):
    """Class to hold estimation results.
    """
    def __init__(self):
        """Initialize the class.
        """
        # Degrees of freedom
        self.df = None
        # J-statistic
        self.jstat = None
        # Optimization results
        self.res = None
        # Standard errors
        self.se = None
        # T-statistics
        self.tstat = None
        # P-values
        self.jpval = None


class GMM(object):
    """GMM estimation class.

    """

    def __init__(self):
        """Initialize the class.
        """
        # initialize class options
        self.__set_default_options()
        # initialize Results instance
        self.results = Results()

    def __set_default_options(self):
        """Set default options.

        """
        # Default options:
        self.options = dict()
        # Number of GMM steps
        self.options['iter'] = 2
        # Optimization method
        self.options['method'] = 'BFGS'
        # Use Jacobian in optimization? Right now it has to be provided anyway.
        self.options['use_jacob'] = True
        # Display convergence results
        self.options['disp'] = False
        # HAC kernel type
        self.options['kernel'] = 'Bartlett'

    def momcond(self, theta):
        """Moment function.

        Computes momcond restrictions and their gradients.
        Should be written for each specific problem.

        Parameters
        ----------
        theta: (k,) array
            Parameters

        Returns
        -------
        g : (T, q) array
            Matrix of momcond restrictions
        dg : (q, k) array
            Gradient of momcond restrictions. Mean over observations

        """
        pass

    def print_results(self):
        """Print GMM estimation results.

        """
        np.set_printoptions(precision=3, suppress=True)

        print('-' * 60)
        print('The final results are')
        print('theta   = ', self.results.theta)
        print('s.e.    = ', self.results.se)
        print('t-stat  = ', self.results.tstat)
        print('J-stat  = %0.2f' % self.results.jstat)
        print('df      = ', self.results.df)
        print('p-value = %0.2f' % self.results.jpval)
        print('-' * 60)

    def gmmest(self, theta_start, **kwargs):
        """Multiple step GMM estimation procedure.

        """
        self.options.update(kwargs)

        # Initialize theta to hold estimator
        theta = theta_start.copy()
        g = self.momcond(theta)[0]
        q = g.shape[1]
        k = len(theta)
        # Number of degrees of freedom
        self.results.df = q - k

        # Weighting matrix
        W = np.eye(q)
        # First step GMM
        for i in range(self.options['iter']):
            # Compute optimal weighting matrix
            # Only after the first step
            if i > 0:
                g = self.momcond(theta)[0]
                W = self.__weights(g)

            output = minimize(self.__gmmobjective, theta, args=(W,),
                              method=self.options['method'],
                              jac=self.options['use_jacob'],
                              callback=self.__callback)
            # Update parameter for the next step
            theta = output.x

        self.results.theta = theta
        # J-statistic
        self.results.jstat = output.fun

        self.descriptive_stat()

    def descriptive_stat(self):
        """Compute descriptive statistics.

        """
        # k x k
        V = self.__varest(self.results.theta)
        # p-value of the J-test, scalar
        self.results.jpval = 1 - chi2.cdf(self.results.jstat, self.results.df)
        # t-stat for each parameter, 1 x k
        self.results.se = np.diag(V)**.5
        # t-stat for each parameter, 1 x k
        self.results.tstat = self.results.theta / self.results.se

    def __callback(self, theta):
        """Callback function. Prints at each optimization iteration."""
        pass

    def __gmmobjective(self, theta, W):
        """GMM objective function and its gradient.

        Parameters
        ----------
        theta : (k,) array
            Parameters
        W : (q, q) array
            Weighting matrix

        Returns
        -------
        f : float
            Value of objective function, see Hansen (2012, p.241)
        df : (k,) array
            Derivative of objective function.
            Depends on the switch 'use_jacob'
        """
        #theta = theta.flatten()
        # g - T x q, time x number of momconds
        # dg - q x k, time x number of momconds
        g, dg = self.momcond(theta)
        T = g.shape[0]
        # g - 1 x q, 1 x number of momconds
        g = g.mean(0)
        gW = g.dot(W)
        f = float(gW.dot(g.T)) * T
        assert f >= 0, 'Objective function should be non-negative'

        if self.options['use_jacob']:
            # 1 x k
            df = 2 * gW.dot(dg) * T
            return f, df
        else:
            return f

    def __weights(self, g):
        """
        Optimal weighting matrix

        Parameters
        ----------
        g : (T, q) array
            Moment restrictions

        Returns
        -------
        (q, q) array
            Inverse of momconds covariance matrix

        """
        return pinv(hac(g, **self.options))

    def __varest(self, theta):
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
        # g - T x q, time x number of momconds
        # dg - q x k, time x number of momconds
        # TODO : What if Jacobian is not returned?
        g, dg = self.momcond(theta)
        # q x q
        S = self.__weights(g)
        # k x k
        # TODO : What if k = 1?
        return pinv(dg.T.dot(S).dot(dg)) / g.shape[0]

if __name__ == '__main__':
    import test_mygmm
    test_mygmm.test_mygmm()
