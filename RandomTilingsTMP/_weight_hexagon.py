from numba import njit
import numpy as np

@njit(["(int64, int64, Array(float64, 2, 'C', False, aligned=True))",
 "(int64, int64, Array(int64, 2, 'C', False, aligned=True))"],cache=True)
def is_there_no_neighbour(x,y,W):
    N = W.shape[0]
    x_ind = N-(y-1)
    if x == 1:
        if y == 1:
            return True
        else:
            return W[x_ind,x-1] == 0
    else:
        if y == 1:
            if ((x+y)%2) == 1:
                return W[x_ind-1,x-2] == 0
            else:
                return W[x_ind-1,x-2] == 0 and W[x_ind-2,x-2] == 0
        elif y == N:
            if ((x+y)%2) == 1:
                return W[x_ind-1,x-2] == 0 and W[x_ind,x-2] == 0 and W[x_ind,x-1] == 0
            else:
                return W[x_ind-1,x-2] == 0 and W[x_ind,x-1] == 0
        else:
            if ((x+y)%2)==1:
                return W[x_ind-1,x-2] == 0 and W[x_ind,x-2] == 0 and W[x_ind,x-1] == 0
            else:
                return W[x_ind-1,x-2] == 0 and W[x_ind-2,x-2] == 0 and W[x_ind,x-1] == 0

@njit("(int64, int64, int64, int64, int64)",cache=True)
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

@njit("(int64, int64, int64, int64, int64)",cache=True)
def isinUpRightCorner(x3,y3,A,B,C):
    return (x3 >= 2*(B+C) and y3 >= 2*(A+C))

@njit("(int64, Array(float64, 2, 'C', False, aligned=True), int64, float64, float64)",cache=True)
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