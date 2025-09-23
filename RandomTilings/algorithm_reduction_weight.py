from RandomTilings.reduction import reduction
from numba import jit
import numpy as np

@jit("(Array(float64, 2, 'C', False, aligned=True),)")
def algorithm_reduction_weight(W):
    n = W.shape[0]//2
    C = n*[np.zeros((0,0))+0j]
    C[n-1] = W + (1+1j)*(W == 0)
    for i in range(2,n+1):
        C[n-i] = reduction(C[n-(i-1)])
    return C