# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:41:32 2012

@author: skhrapov
"""

import numpy as np

def ivmoment(theta, data, options):
    '''
    Moment function, problem specific
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
        
    Output:
        g : T x q, observations x moments
        dg : q x k, gradient mean over observations, moments x parameters
        
    '''
    # 1 x k
    theta = theta.flatten()
    # T x 1
    error = (data['Y'] - np.dot(data['X'], theta)).reshape((data['T'], 1))
    # T x k
    de = - data['X']
    # T x q
    g = error * data['Z']
    # q x k
    dg = (de[:,np.newaxis,:] * data['Z'][:,:,np.newaxis]).mean(0)
    
    return g, dg