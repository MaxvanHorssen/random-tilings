from RandomTilings.probability_matching_2times2 import probability_matching_2times2
from RandomTilings.ids_procedure import ids_procedure
from RandomTilings.creation import creation
from numba import jit
import numpy as np

@jit("(List(Array(complex128, 2, 'C'), True),)")
def shuffling(C):
    if np.random.rand() <= probability_matching_2times2(C[0][0,0],C[0][1,0],C[0][0,1],C[0][1,1]): 
        M = np.array([[1,0],[0,1]])
    else:
        M = np.array([[0,1],[1,0]])
    for i in range(1,len(C)):
        M = ids_procedure(M)
        M = creation(M,C[i]) 
    return M