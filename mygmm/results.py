#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM results class
-----------------

"""
from __future__ import print_function, division

import numpy as np

from scipy.stats import chi2
from functools import partial

__all__ = ['Results']


class Results(object):

    """Class to hold estimation results.

    Attributes
    ----------
    theta
        Parameter estimate
    degf
        Degrees of freedom
    jstat
        J-statistic
    stde
        Standard errors
    tstat
        t-statistics
    jpval
        p-value of the J test
    opt
        Optimization output

    """

    def __init__(self, opt_out=None, var_theta=None, nmoms=None):
        """Initialize the class.

        """
        # Parameter estimate
        self.theta = opt_out.x
        # Degrees of freedom
        self.degf = nmoms - self.theta.size
        # J-statistic
        self.jstat = opt_out.fun
        # Standard errors
        self.stde = np.abs(np.diag(var_theta))**.5
        # t-statistics
        self.tstat = self.theta / self.stde
        # p-value of the J test
        self.jpval = 1 - chi2.cdf(self.jstat, self.degf)
        # Optimization output
        self.opt_out = opt_out

    def __str__(self):
        """Print results of the estimation.

        """
        precision = 4
        suppress = False
        array_str = partial(np.array_str, precision=precision,
                            suppress_small=suppress)
        show = '-' * 60
        show += '\nThe final results are'
        show += '\ntheta   = ' + array_str(self.theta)
        show += '\ns.e.    = ' + array_str(self.stde)
        show += '\nt-stat  = ' + array_str(self.tstat)
        show += '\nJ-stat  = %0.2f' % self.jstat
        show += '\ndf      = ' + str(self.degf)
        show += '\np-value = %0.2f' % self.jpval
        show += '\n' + '-' * 60
        return show

    def __repr__(self):
        """String representation.

        """
        repr = self.__str__()
        repr.replace('\n', '')
        return repr
