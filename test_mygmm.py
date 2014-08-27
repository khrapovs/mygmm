# To use the module from a different location, do:
# import sys
# sys.path.append("~/Dropbox/Computation/Python/MyGMM")

import numpy as np
from gmm import GMM
import pandas as pd

def ivmoment(theta, data, options):
    """Moment function, problem specific.
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
        
    Output:
        g : T x q, observations x moments
        dg : q x k, gradient mean over observations, moments x parameters
        
    """
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

def test_mygmm():
    # Number of observations
    T = 1e6
    # Correlation
    rho = .9
    # True parameter
    beta = np.array([1., -.5])
    # True parameters for instruments
    gamma = np.array([1, -5, 2, 3, -1])
    # Random errors
    e = np.random.normal(size = (T, 2))
    # Instruments
    Z = np.random.normal(size = (T, 5))
    # Endogenous variables
    X1 = np.dot(Z, gamma) + e[:,0]
    X2 = np.dot(Z**2, gamma) + e[:,1]
    X = np.concatenate((X1[:, np.newaxis], X2[:, np.newaxis]), axis = 1)
    # Dependent variable
    Y = np.dot(X, beta) + e[:,0] + rho * e[:,1]
    
    #plt.scatter(X, Y)
    #plt.show()
    
    # Collect data for GMM
    data = {'T' : Y.shape[0], 'q' : Z.shape[1], 'k' : beta.shape[0],
            'Y' : Y, 'X' : X, 'Z' : Z}
    
    # Initialize options for GMM
    options = {'moment' : ivmoment,
               'W' : np.eye(Z.shape[1]),
               'Iter' : 2,
               'tol' : None,
               'maxiter' : 10,
               'method' : 'BFGS',
               'disp' : True,
               'precision' : 3,
               'jacob' : True,
               'kernel' : 'Bartlett',
               'bounds' : None,
               'band' : int(data['T']**(1/3))}
    
    # Initialize GMM object
    gmm = GMM(beta*2, data, options)
    # Estimate model with GMM
    gmm.gmmest()
    # Print results
    gmm.results()
    
    # Compare with OLS
    Xps = pd.DataFrame(X)
    Xps = Xps.rename(columns = {0 : 'X1', 1 : 'X2'})
    Yps = pd.DataFrame(Y)
    Yps = Yps.rename(columns = {0 : 'Y'})
    df = pd.merge(Yps, Xps, left_index = True, right_index = True)
    res = pd.ols(y = df['Y'], x = df[['X1','X2']], intercept = False)
    print(np.array(res.beta))
    print(np.array(res.t_stat))

if __name__ == '__main__':
    test_mygmm()