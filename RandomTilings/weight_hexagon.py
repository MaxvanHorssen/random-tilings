from RandomTilings.is_there_no_neighbour import is_there_no_neighbour
from numba import jit
import numpy as np

@jit() #("(int64, int64, int64, int64, int64)")
def isinhexagon(x2,y2,A,B,C):
    if x2 <= 2*min(B,C):
        return y2 >= 1 and y2 <= 2*A+x2-1
    elif x2 >= 2*C+1 and x2 <= 2*B:
        return y2 >= 1 and y2 <= 2*(A+C)-1
    elif x2 >= 2*B+1 and x2 <= 2*C:
        return y2 >= x2-2*B+1 and y2 <= 2*A+x2-1
    elif x2 >= 2*max(B,C)+1 and x2 <= 2*(B+C)-1:
        return y2 >= x2-2*B+1 and y2 <= 2*(A+C)-1
    else:
        return False

@jit() #("(int64, int64, int64, int64, int64)")
def isinUpRightCorner(x3,y3,A,B,C):
    return (x3 >= 2*(B+C) and y3 >= 2*(A+C))

@jit("(int64, Array(float64, 2, 'C', False, aligned=True), int64, float64, float64)")
def weight_hexagon(n,w,a,b,c):
    A = int(np.round(a*n))
    B = int(np.round(b*n))
    C = int(np.round(c*n))
    N = 2*(A+B+C-1)

    W = np.zeros((N,N))
    Lp_space,Lp_time = w.shape[0],w.shape[1]

    for x in range(1,N+1):
        for y in range(1,N+1):
            if isinhexagon(x,y,A,B,C):
                if x%2 == 1:
                    k = (x+1)//2
                    W[N-y,x-1] = w[Lp_space-1-((y-1)%Lp_space),(k-1)%Lp_time]
                elif y%2 == 1:
                    W[N-y,x-1] = 1
            elif isinUpRightCorner(x,y,A,B,C) and is_there_no_neighbour(x,y,W):
                W[N-y,x-1] = 1
            elif is_there_no_neighbour(x,y,W) and (y+x)%2 == 1:
                W[N-y,x-1] = 1
    return W