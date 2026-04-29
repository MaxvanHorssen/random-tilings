from numpy import zeros,ascontiguousarray,isinf
from numba import njit
from numba.typed import List
from numpy import save
from tqdm import tqdm
import os

@njit("(float64, float64, float64, float64)")
def red_cases_tmp(w,x,y,z): 
    if w == 0:
        tmp_w = 0
    elif z == 0:
        tmp_w = w/(x*y)
    elif x*y == 0:
        tmp_w = 1/z
    elif isinf(w): 
        tmp_w = 1/z
    elif isinf(x*y) or isinf(z):
        tmp_w = 0
    else:
        tmp_w = w/(w*z+x*y)
    return tmp_w





@njit("(float64, float64, float64, float64)")
def red_cases(w,x,y,z):
    if w+x+y+z == 0:
        return 1/2**0.5,1/2**0.5,1/2**0.5,1/2**0.5
    elif z+x==0:
        return 1/(w+y),y,1/(w+y),w
    elif w+x==0:
        return z,y,1/(z+y),1/(z+y)
    elif w+y==0:
        return z,1/(z+x),x,1/(z+x)
    elif z+y==0:
        return 1/(x+w),1/(x+w),x,w
    else:
        tmp_w = red_cases_tmp(w,x,y,z)
        tmp_x = red_cases_tmp(x,w,z,y) # w <-> x, y <-> z
        tmp_y = red_cases_tmp(y,z,w,x) # w <-> y, x <-> z
        tmp_z = red_cases_tmp(z,x,y,w) # w <-> z
        return tmp_z, tmp_y, tmp_x, tmp_w


@njit(["(Array(float64, 2, 'C', False, aligned=True),)","(Array(float64, 2, 'A', False, aligned=True),)"])
def reduction(W):
    N = W.shape[0]
    W_red = zeros((N,N))

    for m in range(N//2):
        for n in range(N//2):
            w,x,y,z = W[2*m,2*n],W[2*m,2*n+1],W[2*m+1,2*n],W[2*m+1,2*n+1]
            p,q,r,s = red_cases(w,x,y,z)
            W_red[2*m,2*n]     = p
            W_red[2*m,2*n+1]   = q
            W_red[2*m+1,2*n]   = r
            W_red[2*m+1,2*n+1] = s

    for m in range(N//2):
        for n in range(N//2):
            w,x,y,z = W[2*m,2*n],W[2*m,2*n+1],W[2*m+1,2*n],W[2*m+1,2*n+1]
            if w+x == 0:
                W_red[2*m-1,2*n],W_red[2*m-1,2*n+1]=0,0
            if x+z == 0:
                W_red[2*m,2*n+2],W_red[2*m+1,2*n+2]=0,0
            if y+z == 0:
                W_red[2*m+2,2*n],W_red[2*m+2,2*n+1]=0,0
            if w+y == 0:
                W_red[2*m,2*n-1],W_red[2*m+1,2*n-1]=0,0
        
                
    return W_red[1:-1,1:-1]



@njit("(Array(float64, 2, 'C', False, aligned=True),)")
def reduce_weight(W):
    n = W.shape[0] // 2
    C = List()
    C.append(ascontiguousarray(W))
    for _ in range(2, n + 1):
        R = reduction(C[-1])
        C.append(ascontiguousarray(R))
    return C[::-1]


def clear_CTower():
    folder_path = "CTower"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            os.remove(file_path)


def reduce_weight_hd(W):
    clear_CTower()
    N = W.shape[0] // 2
    C = W
    save('CTower/'+str(N-1),C)
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    for k in tqdm(range(N-2, -1,-1),desc="Reduction",bar_format=bar_format):
        C = reduction(C)
        save('CTower/'+str(k),C)
    return


@njit("(Array(float64, 2, 'C', False, aligned=True),)")
def reduce_weight_memory(W):
    K = W.shape[0] // 2
    C = W.copy()
    for k in range(K-1):
        for m in range(K-k):
            for n in range(K-k):
                d = C[k+2*m,k+2*n]*C[k+2*m+1,k+2*n+1]+C[k+2*m+1,k+2*n]*C[k+2*m,k+2*n+1]
                w,x,y,z = C[k+2*m,k+2*n],C[k+2*m,k+2*n+1],C[k+2*m+1,k+2*n],C[k+2*m+1,k+2*n+1]
                C[k+2*m,k+2*n]     = z/d
                C[k+2*m,k+2*n+1]   = y/d
                C[k+2*m+1,k+2*n]   = x/d
                C[k+2*m+1,k+2*n+1] = w/d
    return C