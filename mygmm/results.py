#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM results class
-----------------

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd

from scipy.stats import chi2

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

    def __init__(self, opt_out=None, var_theta=None, nmoms=None, names=None):
        """Initialize the class.

        """
        # Parameter estimate
        self.theta = opt_out.x
        # Parameter names
        self.names = names
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
        cols = {'theta': self.theta, 'std': self.stde, 'tstat': self.tstat}
        res = pd.DataFrame(cols, index=self.names)[['theta', 'std', 'tstat']]
        res_str = res.to_string(float_format=lambda x: '%.4f' % x)
        width = len(res_str) // (res.shape[0] + 1)
        show = '-' * 60
        show += '\nGMM final results:\n'
        show += width * '-' + '\n'
        show += res_str
        show += '\n' + width * '-'
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
