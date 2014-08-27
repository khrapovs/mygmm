import numpy as np
from scipy import optimize, stats, linalg
from hac import hac
#from numba import autojit

def results(gmm, data, options):
    """Print results function.
    
    Does not return anything. Just prints.
    """
    np.set_printoptions(precision = options['precision'], suppress = True)
    
    print('-' * 60)
    print('The final results are')
    print(gmm.message)
    print('theta   = ', gmm.x)
    print('s.e.    = ', gmm.se)
    print('t-stat  = ', gmm.t)
    print('J-stat  = ', '%0.2f' % gmm.fun)
    print('df      = ', gmm.df)
    print('p-value = ', '%0.2f' % gmm.pval)
    print('-' * 60)


def gmmest(theta, data, options):
    """Multiple step GMM estimation procedure.
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
        
    Output:
        gmm : object containing optimization results and statistics
        
    """
    print('Theta 0 = ', theta)
    # First step GMM
    for i in range(options['Iter']):
        # Compute optimal weighting matrix
        # Only after the first step
        if i > 0:
            options['W'] = weights(theta, data, options)
        # i step GMM
        gmm = optimize.minimize(gmmobjective, theta, (data, options), 
                                method = options['method'],
                                jac = options['jacob'],
                                tol = options['tol'],
                                bounds = options['bounds'],
                                options = {'disp' : options['disp'],
                                           'maxiter' : options['maxiter']})
        # Update parameter for the next step
        theta = gmm.x
        gmm.fun = data['T'] * gmm.fun
        print('Theta', i+1, ' = ', gmm.x)
        print('f', i+1, ' = ', gmm.fun)
    
    gmm.W = options['W']
    # k x k
    V = varest(gmm.x, data, options)
    # degrees of freedom, scalar
    gmm.df = data['q'] - data['k']
    # p-value of the J-test, scalar
    gmm.pval = 1 - stats.chi2.cdf(gmm.fun, gmm.df)
    # t-stat for each parameter, 1 x k
    gmm.se = np.sqrt(np.diag(V))
    # t-stat for each parameter, 1 x k
    gmm.t = gmm.x / gmm.se
    
    return gmm

#@autojit
def gmmobjective(theta, data, options):
    """GMM objective function and its gradient.
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
    
    Output:
        f : 1 x 1, value of objective function, see Hansen (2012, p.241)
        df : 1 x k, derivative of objective function, 1 x parameters
    """
    #theta = theta.flatten()
    # g - T x q, time x number of moments
    # dg - q x k, time x number of moments
    g, dg = options['moment'](theta, data, options)
    # g - 1 x q, 1 x number of moments
    g = g.mean(0).flatten()
    # 1 x 1
    f = float(np.dot(np.dot(g, options['W']), g.T))
    
    if options['jacob']:
        # 1 x k    
        df = 2 * np.dot(np.dot(g, options['W']), dg).flatten()
        
        return f, df
    else:
        return f


def weights(theta, data, options):
    """
    Optimal weighting matrix
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
        
    Output:
        invS : q x q, moments x moments
        
    """
    # g - T x q, time x number of moments
    # dg - q x k, time x number of moments
    g, dg = options['moment'](theta, data, options)
    # q x q
    S = hac(g, options['kernel'], options['band'])
    # q x q
    invS = linalg.pinv(S)
    
    return invS

    
def varest(theta, data, options):
    """Variance matrix of parameters.
    
    Inputs:
        theta : vector, 1 x k
        data : problem scpecific
        options : control of optimization, etc.
        
    Output:
        V : k x k, parameters x parameters
        
    """
    # g - T x q, time x number of moments
    # dg - q x k, time x number of moments
    g, dg = options['moment'](theta, data, options)
    # q x q
    S = weights(theta, data, options)
    # k x k
    V = linalg.pinv(np.atleast_2d(np.dot(np.dot(dg.T, S), dg))) / data['T']
    
    return V



if __name__ == '__main__':
    import test_mygmm
    test_mygmm.test_mygmm()