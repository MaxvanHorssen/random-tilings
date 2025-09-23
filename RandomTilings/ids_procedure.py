from numba import jit
import numpy as np

@jit("(Array(int64, 2, 'C', False, aligned=True),)")
def ids_procedure(M):
    # Inclusion
    N = M.shape[0]
    M_bigger = np.array((N+2)*[(N+2)*[0]])
    M_bigger[1:N+1,1:N+1] = M

    # Destruction
    for x in range(1,N+2,2):
        for y in range(1,N+2,2):
            if np.count_nonzero(M_bigger[N+1-y:N+3-y,x-1:x+1]) == 2:
                M_bigger[N+1-y:N+3-y,x-1:x+1] = np.array([[0,0],[0,0]])

    # Sliding
    for x in range(1,N+2,2):
        for y in range(1,N+2,2):
            if np.count_nonzero(M_bigger[N+1-y:N+3-y,x-1:x+1]) == 1:
                M_bigger[N+1-y:N+3-y,x-1:x+1] = np.array([[M_bigger[N+2-y,x], M_bigger[N+2-y,x-1]],
                                                   [M_bigger[N+1-y,x], M_bigger[N+1-y,x-1]]])
    return M_bigger