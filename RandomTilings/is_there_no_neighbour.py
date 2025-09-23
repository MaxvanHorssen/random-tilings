from numba import jit

@jit(["(int64, int64, Array(float64, 2, 'C', False, aligned=True))",
 "(int64, int64, Array(int64, 2, 'C', False, aligned=True))"])
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