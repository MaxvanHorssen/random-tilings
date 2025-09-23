from RandomTilings.probability_matching_2times2 import probability_matching_2times2
from RandomTilings.is_there_no_neighbour import is_there_no_neighbour
from numba import jit
import numpy as np

@jit(["(Array(int64, 2, 'C', False, aligned=True),Array(complex128, 2, 'C', False, aligned=True))"])
def creation(M,W):
    N = M.shape[0]
    for x in range(1,N,2):
        for y in range(1,N,2):
            if (is_there_no_neighbour(x,y,M) and is_there_no_neighbour(x,y+1,M) 
                and is_there_no_neighbour(x+1,y,M) and is_there_no_neighbour(x+1,y+1,M)):
                if 0 in M[N-y-1:N-y+1,x-1:x+1]:
                    if np.random.rand() <= probability_matching_2times2(W[N-y-1,x-1],
                                                                        W[N-y,x-1],W[N-y-1,x],W[N-y,x]):
                        M[N-y-1:N-y+1,x-1:x+1] = np.array([[1,0],[0,1]])
                    else:
                        M[N-y-1:N-y+1,x-1:x+1] = np.array([[0,1],[1,0]])
    return M