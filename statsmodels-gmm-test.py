import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
import matplotlib.pylab as plt

def generate_data():
    # Number of observations
    T = 1e3
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
    
    print(X.shape, Y.shape)
    
    plt.scatter(X[:,0], Y)
    plt.show()
    
    return Y, X, Z

class my_GMM(GMM):
    def momcond(self, params):
        # 1 x k
        endog = self.endog
        exog = self.exog
        instrument = self.instrument
        # T x 1
        error = endog - np.dot(exog, params)
        g = error[:,np.newaxis] * instrument
        return g

def test_gmm():
    Y, X, Z = generate_data()
    model = my_GMM(Y, X, Z)
    beta0 = np.array([1., -.5]) * 10
    res = model.fititer(beta0, maxiter = 100)
    beta_hat = res[0]
    
    print(beta_hat)

if __name__ == '__main__':
    test_gmm()