# To use the module from a different location, do:
# import sys
# sys.path.append("~/Dropbox/Computation/Python/MyGMM")

import numpy as np
from gmm import GMM
import pandas as pd

class Model(GMM):
    """Model moment restrictions and Jacobian.
    
    Inherits from GMM class.    
    """
    def __init__(self, theta, data):
        super(Model, self).__init__(theta, data)
    
    def moment(self, theta):
        return moment(theta, self.data)

def moment(theta, data):
    """Moment function, problem specific.
    
    Args:
        theta : vector, 1 x k
        
    Returns:
        g : T x q, matrix of moment restrictions
        dg : q x k, gradient of moment restrictions. Mean over observations
    """
    # T x 1
    error = data['Y'] - data['X'].dot(theta)
    # T x k
    de = -data['X']
    # T x q
    g = (error * data['Z'].T).T
    # q x k
    dg = (de[:,np.newaxis,:] * data['Z'][:,:,np.newaxis]).mean(0)
    
    return g, dg


def simulate_data():
    """Simulate data.
    
    Args:
        None
        
    Returns:
        data: disctionary of model specific data arrays
        beta: true parameter
    """
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
    X1 = Z.dot(gamma) + e[:,0]
    X2 = (Z**2).dot(gamma) + e[:,1]
    X = np.concatenate((X1[:, np.newaxis], X2[:, np.newaxis]), axis = 1)
    # Dependent variable
    Y = X.dot(beta) + e[:,0] + rho * e[:,1]
    
    #plt.scatter(X, Y)
    #plt.show()
    
    # Collect data for GMM
    data = {'Y' : Y, 'X' : X, 'Z' : Z}
    
    return data, beta
    
def test_mygmm():
    
    data, theta_true = simulate_data()
    # Initialize GMM object
    model = Model(theta_true*2, data)
    # Estimate model with GMM
    model.gmmest()
    # Print results
    model.print_results()
    
    # Compare with OLS
    df = pd.DataFrame(data['X'], columns = ['X1','X2'])
    df['Y'] = data['Y']
    res = pd.ols(y = df['Y'], x = df[['X1','X2']], intercept = False)
    print('\nOLS results')
    print(np.array(res.beta))
    print(np.array(res.t_stat))

if __name__ == '__main__':
    test_mygmm()