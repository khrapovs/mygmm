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
        """Moment function, problem specific.
        
        Args:
            theta : vector, 1 x k
            data : problem scpecific
            options : control of optimization, etc.
            
        Returns:
            g : T x q, observations x moments
            dg : q x k, gradient mean over observations, moments x parameters
            
        """
        # T x 1
        error = self.data['Y'] - self.data['X'].dot(theta)
        # T x k
        de = - self.data['X']
        # T x q
        g = (error * self.data['Z'].T).T
        # q x k
        dg = (de[:,np.newaxis,:] * self.data['Z'][:,:,np.newaxis]).mean(0)
        
        return g, dg

def simulate_data():
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
    Xps = pd.DataFrame(data['X'])
    Xps = Xps.rename(columns = {0 : 'X1', 1 : 'X2'})
    Yps = pd.DataFrame(data['Y'])
    Yps = Yps.rename(columns = {0 : 'Y'})
    df = pd.merge(Yps, Xps, left_index = True, right_index = True)
    res = pd.ols(y = df['Y'], x = df[['X1','X2']], intercept = False)
    print('\nOLS results')
    print(np.array(res.beta))
    print(np.array(res.t_stat))

if __name__ == '__main__':
    test_mygmm()