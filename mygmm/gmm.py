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

    def gmmest(self, theta_start, bounds=None, iter=2, method='BFGS',
               kernel='Bartlett', band=None, **kwargs):
        """Multiple step GMM estimation procedure.

        Parameters
        ----------
        theta_start : array
            Initial parameters
        bounds : list of tuples
            Bounds on parameters
        iter : int
            Number of GMM steps
        method : str
            Optimization method
        kernel : str
            Type of kernel for HAC.
            Currenly implemented: SU, Bartlett, Parzen, Quadratic
        band: int
            Truncation parameter for HAC

        Returns
        -------
        instance of Results
            Estimation results

        """
        # Initialize theta to hold estimator
        theta = theta_start.copy()

        # First step GMM
        for i in range(iter):
            moment = self.momcond(theta, **kwargs)[0]
            nmoms = moment.shape[1]
            if nmoms - theta.size <= 0:
                warnings.warn("Not enough degrees of freedom!")
            nmoms = moment.shape[1]
            # Compute optimal weighting matrix
            # Only after the first step
            if i == 0:
                weight_mat = np.eye(nmoms)
            else:
                weight_mat = self.__weights(moment, kernel=kernel, band=band)

            opt_out = minimize(self.__gmmobjective, theta,
                              args=(weight_mat, kwargs), method=method,
                              jac=True, bounds=bounds,
                              callback=self.callback)
            # Update parameter for the next step
            theta = opt_out.x

        var_theta = self.varest(theta, **kwargs)

        return Results(opt_out=opt_out, var_theta=var_theta, nmoms=nmoms)

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

        if dmoment is None:
            dmoment = self.approx_dmoment(theta, **kwargs)
        # 1 x nparams
        dvalue = 2 * gdotw.dot(dmoment) * nobs
        return value, dvalue

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
