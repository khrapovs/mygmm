#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GMM estimator.

"""

from __future__ import print_function, division

import numpy as np
from scipy.linalg import pinv
from scipy.stats import chi2
from scipy.optimize import minimize

from .hac_function import hac

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"

__all__ = ['GMM', 'Results']


class Results(object):

    """Class to hold estimation results.

    """

    def __init__(self):
        """Initialize the class.
        """
        # Parameter estimate
        self.theta = None
        # Degrees of freedom
        self.degf = None
        # J-statistic
        self.jstat = None
        # Standard errors
        self.stde = None
        # t-statistics
        self.tstat = None
        # p-value of the J test
        self.jpval = None

    def print_results(self):
        """Print results of the estimation.
        """
        print('-' * 60)
        print('The final results are')
        print('theta   = ', self.theta)
        print('s.e.    = ', self.stde)
        print('t-stat  = ', self.tstat)
        print('J-stat  = %0.2f' % self.jstat)
        print('df      = ', self.degf)
        print('p-value = %0.2f' % self.jpval)
        print('-' * 60)


class GMM(object):

    """GMM estimation class.

    Attributes
    ----------
    momcond

    Methods
    -------
    gmmest

    """

    def __init__(self, momcond):
        """Initialize the class.
        """
        # Moment conditions
        self.momcond = momcond
        # Default options:
        self.options = dict()
        # initialize class options
        self.__set_default_options()
        # initialize Results instance
        self.results = Results()

    def __set_default_options(self):
        """Set default options.

        """
        # Number of GMM steps
        self.options['iter'] = 2
        # Optimization method
        self.options['method'] = 'BFGS'
        # Use Jacobian in optimization? Right now it has to be provided anyway.
        self.options['use_jacob'] = True
        # HAC kernel type
        self.options['kernel'] = 'Bartlett'

    def gmmest(self, theta_start, **kwargs):
        """Multiple step GMM estimation procedure.

        """
        self.options.update(kwargs)

        # Initialize theta to hold estimator
        theta = theta_start.copy()

        # First step GMM
        for i in range(self.options['iter']):
            moment = self.momcond(theta, **kwargs)[0]
            nmoms = moment.shape[1]
            # Compute optimal weighting matrix
            # Only after the first step
            if i == 0:
                weight_mat = np.eye(nmoms)
            else:
                weight_mat = self.__weights(moment, **kwargs)

            output = minimize(self.__gmmobjective, theta,
                              args=(weight_mat, kwargs),
                              method=self.options['method'],
                              jac=self.options['use_jacob'],
                              callback=self.callback)
            # Update parameter for the next step
            theta = output.x

        self.__descriptive_stat(output, weight_mat, **kwargs)
        return self.results

    def __descriptive_stat(self, output, weight_mat, **kwargs):
        """Compute descriptive statistics.

        """
        # Final theta
        self.results.theta = output.x
        # Number of degrees of freedom
        self.results.degf = weight_mat.shape[0] - output.x.shape[0]
        # J-statistic
        self.results.jstat = output.fun
        # Variance matrix of parameters
        var_theta = self.__varest(self.results.theta, **kwargs)
        # p-value of the J-test, scalar
        self.results.jpval = 1 - chi2.cdf(self.results.jstat,
                                          self.results.degf)
        # t-stat for each parameter, 1 x k
        self.results.stde = np.diag(var_theta)**.5
        # t-stat for each parameter, 1 x k
        self.results.tstat = self.results.theta / self.results.stde

    def callback(self, theta):
        """Callback function. Prints at each optimization iteration."""
        pass

    def __gmmobjective(self, theta, weight_mat, kwargs):
        """GMM objective function and its gradient.

        Parameters
        ----------
        theta : (nparams,) array
            Parameters
        weight_mat : (nmoms, nmoms) array
            Weighting matrix

        Returns
        -------
        value : float
            Value of objective function, see Hansen (2012, p.241)
        dvalue : (nparams,) array
            Derivative of objective function.
            Depends on the switch 'use_jacob'
        """
        # moment - nobs x nmoms
        # dmoment - nmoms x nparams
        moment, dmoment = self.momcond(theta, **kwargs)
        nobs = moment.shape[0]
        moment = moment.mean(0)
        gdotw = moment.dot(weight_mat)
        # Objective function
        value = gdotw.dot(moment.T) * nobs
        if value <= 0:
            value = 1e10
        #assert value >= 0, 'Objective function should be non-negative'

        if self.options['use_jacob']:
            # 1 x nparams
            dvalue = 2 * gdotw.dot(dmoment) * nobs
            return value, dvalue
        else:
            return value

    def __weights(self, moment, **kwargs):
        """
        Optimal weighting matrix

        Parameters
        ----------
        moment : (nobs, nmoms) array
            Moment restrictions

        Returns
        -------
        (nmoms, nmoms) array
            Inverse of momconds covariance matrix

        """
        return pinv(hac(moment, **self.options))

    def __varest(self, theta, **kwargs):
        """Estimate variance matrix of parameters.

        Parameters
        ----------
        theta : (nparams,)
            Parameters

        Returns
        -------
        (nparams, nparams) array
            Variance matrix of parameters

        """
        # g - nobs x q, time x number of momconds
        # dmoment - q x k, time x number of momconds
        # TODO : What if Jacobian is not returned?
        moment, dmoment = self.momcond(theta, **kwargs)
        var_moment = self.__weights(moment)
        # TODO : What if k = 1?
        return pinv(dmoment.T.dot(var_moment).dot(dmoment)) / moment.shape[0]


if __name__ == '__main__':
    pass
