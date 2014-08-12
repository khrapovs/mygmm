import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
import matplotlib.pylab as plt

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

model = my_GMM(Y, X, Z)
res = model.fititer(beta, maxiter = 100)

print(res[0])