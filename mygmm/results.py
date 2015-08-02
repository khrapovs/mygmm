#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMM results class
-----------------

"""
from __future__ import print_function, division

import numpy as np

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
        # Optimization output
        self.opt = None

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
