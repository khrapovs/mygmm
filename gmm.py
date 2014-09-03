import numpy as np
from scipy import optimize, stats, linalg
from hac import hac
#from numba import autojit

class GMM(object):
    
    def __init__(self, theta_init, data, options):
        self.data = data
        self.theta = theta_init
        self.options = options
        
    def results(self):
        """Print results function.
        
        Does not return anything. Just prints.
        """
        np.set_printoptions(precision = self.options['precision'],
                            suppress = True)
        
        print('-' * 60)
        print('The final results are')
        print(self.res.message)
        print('theta   = ', self.res.x)
        print('s.e.    = ', self.res.se)
        print('t-stat  = ', self.res.t)
        print('J-stat  = ', '%0.2f' % self.res.fun)
        print('df      = ', self.res.df)
        print('p-value = ', '%0.2f' % self.res.pval)
        print('-' * 60)
    
    
    def gmmest(self):
        """Multiple step GMM estimation procedure.
        
        Args:
            theta : vector, 1 x k
            data : problem scpecific
            options : control of optimization, etc.
            
        Returns:
            gmm : object containing optimization results and statistics
            
        """
        print('Theta 0 = ', self.theta)
        # First step GMM
        for i in range(self.options['Iter']):
            # Compute optimal weighting matrix
            # Only after the first step
            if i > 0:
                self.options['W'] = self.weights()
            # i step GMM
            
            opt_options = {'disp' : self.options['disp'],
                           'maxiter' : self.options['maxiter']}
            self.res = optimize.minimize(self.gmmobjective, self.theta,
                                    method = self.options['method'],
                                    jac = self.options['jacob'],
                                    tol = self.options['tol'],
                                    bounds = self.options['bounds'],
                                    options = opt_options)
            # Update parameter for the next step
            self.theta = self.res.x
            self.res.fun = self.data['T'] * self.res.fun
            print('Theta', i+1, ' = ', self.res.x)
            print('f', i+1, ' = ', self.res.fun)
        
        self.res.W = self.options['W']
        # k x k
        V = self.varest()
        # degrees of freedom, scalar
        self.res.df = self.data['q'] - self.data['k']
        # p-value of the J-test, scalar
        self.res.pval = 1 - stats.chi2.cdf(self.res.fun, self.res.df)
        # t-stat for each parameter, 1 x k
        self.res.se = np.diag(V)**.5
        # t-stat for each parameter, 1 x k
        self.res.t = self.res.x / self.res.se
    
    def gmmobjective(self, theta):
        """GMM objective function and its gradient.
        
        Args:
            theta : vector, 1 x k
            data : problem scpecific
            options : control of optimization, etc.
        
        Returns:
            f : 1 x 1, value of objective function, see Hansen (2012, p.241)
            df : 1 x k, derivative of objective function, 1 x parameters
        """
        #theta = theta.flatten()
        # g - T x q, time x number of moments
        # dg - q x k, time x number of moments
        g, dg = self.options['moment'](theta, self.data, self.options)
        # g - 1 x q, 1 x number of moments
        g = g.mean(0).flatten()
        # 1 x 1
        f = float(g.dot(self.options['W']).dot(g.T))
        
        if self.options['jacob']:
            # 1 x k    
            df = 2 * g.dot(self.options['W']).dot(dg).flatten()
            
            return f, df
        else:
            return f
    
    def weights(self):
        """
        Optimal weighting matrix
        
        Args:
            theta : vector, 1 x k
            data : problem scpecific
            options : control of optimization, etc.
            
        Returns:
            invS : q x q, moments x moments
            
        """
        # g - T x q, time x number of moments
        # dg - q x k, time x number of moments
        g, dg = self.options['moment'](self.theta, self.data, self.options)
        # q x q
        S = hac(g, self.options['kernel'], self.options['band'])
        # q x q
        invS = linalg.pinv(S)
        
        return invS
    
    def varest(self):
        """Variance matrix of parameters.
        
        Args:
            theta : vector, 1 x k
            data : problem scpecific
            options : control of optimization, etc.
            
        Returns:
            V : k x k, parameters x parameters
            
        """
        # g - T x q, time x number of moments
        # dg - q x k, time x number of moments
        g, dg = self.options['moment'](self.theta, self.data, self.options)
        # q x q
        S = self.weights()
        # k x k
        V = linalg.pinv(np.atleast_2d(dg.T.dot(S).dot(dg))) / self.data['T']
        
        return V

if __name__ == '__main__':
    import test_mygmm
    test_mygmm.test_mygmm()