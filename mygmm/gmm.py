#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM estimator class
-------------------

"""
from __future__ import print_function, division

import warnings

import numpy as np
import numdifftools as nd

from scipy.linalg import pinv
from scipy.stats import chi2
from scipy.optimize import minimize

from .hac_function import hac
from .results import Results

__all__ = ['GMM']


class GMM(object):

    """GMM estimation class.

    Attributes
    ----------
    momcond
        Moment function

    Methods
    -------
    gmmest
        Multiple step GMM estimation procedure

    """

    def __init__(self, momcond):
        """Initialize the class.

        Parameters
        ----------
        momcond : function
            Moment function. Should return:

                - array (nobs x nmoms)
                    moment function values
                - (optionally) array (nmoms x nparams)
                    derivative of moment function average across observations.

        """
        # Moment conditions
        self.momcond = momcond
        self.use_jacob = True
        # initialize Results instance
        self.results = Results()

    def gmmest(self, theta_start, bounds=None, iter=2, method='BFGS',
               use_jacob=True, kernel='Bartlett', band=None, **kwargs):
        """Multiple step GMM estimation procedure.

        Parameters
        ----------
        theta_start : array
            Initial parameters

        Returns
        -------
        instance of Results
            Estimation results

        """
        self.use_jacob = use_jacob
        # Initialize theta to hold estimator
        theta = theta_start.copy()

        # First step GMM
        for i in range(iter):
            moment = self.momcond(theta, **kwargs)[0]
            if moment.shape[1] <= theta.size:
                warnings.warn("Not enough degrees of freedom!")
            nmoms = moment.shape[1]
            # Compute optimal weighting matrix
            # Only after the first step
            if i == 0:
                weight_mat = np.eye(nmoms)
            else:
                weight_mat = self.__weights(moment, kernel=kernel, band=band)

            output = minimize(self.__gmmobjective, theta,
                              args=(weight_mat, kwargs), method=method,
                              jac=self.use_jacob, bounds=bounds,
                              callback=self.callback)
            # Update parameter for the next step
            theta = output.x

        self.__descriptive_stat(output, weight_mat, **kwargs)
        self.results.opt = output
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
        var_theta = self.varest(self.results.theta, **kwargs)
        # p-value of the J-test, scalar
        self.results.jpval = 1 - chi2.cdf(self.results.jstat,
                                          self.results.degf)
        # t-stat for each parameter, 1 x k
        self.results.stde = np.abs(np.diag(var_theta))**.5
        # t-stat for each parameter, 1 x k
        self.results.tstat = self.results.theta / self.results.stde

    def callback(self, theta):
        """Callback function. Prints at each optimization iteration.

        """
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
        # assert value >= 0, 'Objective function should be non-negative'

        if self.use_jacob:
            if dmoment is None:
                dmoment = self.approx_dmoment(theta, **kwargs)
            # 1 x nparams
            dvalue = 2 * gdotw.dot(dmoment) * nobs
            return value, dvalue
        else:
            return value

    def approx_dmoment(self, theta, **kwargs):
        """Approxiamte derivative of the moment function numerically.

        Parameters
        ----------
        theta : (nparams,) array
            Parameters

        Returns
        -------
        (nmoms, nparams) array
            Derivative of the moment function

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(lambda x:
                self.momcond(x, **kwargs)[0].mean(0))(theta)

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
        return pinv(hac(moment, **kwargs))

    def varest(self, theta, **kwargs):
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
        moment, dmoment = self.momcond(theta, **kwargs)
        if dmoment is None:
            dmoment = self.approx_dmoment(theta, **kwargs)
        var_moment = self.__weights(moment, **kwargs)
        # TODO : What if k = 1?
        return pinv(dmoment.T.dot(var_moment).dot(dmoment)) / moment.shape[0]
