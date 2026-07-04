import numpy as np
from numba import jit

@jit("(int64, Array(float64, 2, 'C', False, aligned=True))",cache=True)
def weight_aztec(n,w):
    N = 2*n
    W = np.zeros((N,N))
    end = N
    y_period,x_period = w.shape[0],w.shape[1]

    for x in range(1,N+1):
        for y in range(1,N+1):
            W[end-y,x-1] = w[y_period-((y-1)%y_period)-1,(x-1)%x_period]
    return W

@jit("(int64, Array(complex128, 2, 'C', False, aligned=True))",cache=True)
def weight_aztec_forceE(n,w):
    N = 2*n
    W = np.zeros((N,N))+0j
    end = N
    y_period,x_period = w.shape[0],w.shape[1]

    for x in range(1,N+1):
        for y in range(1,N+1):
            W[end-y,x-1] = w[y_period-((y-1)%y_period)-1,(x-1)%x_period]
    return W