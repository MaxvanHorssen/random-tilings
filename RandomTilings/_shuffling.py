from numpy import zeros,int8,load
from numpy.random import random
from numba import njit
from tqdm import tqdm

@njit( "(Array(int8, 2, 'C', False, aligned=True), int64, int64, int64)")
def no_neighbor(M,k,m,n):
    N = M.shape[0]
    A = N//2-1-k  
    
    if M[A+2*m,A+2*n]+M[A+2*m+1,A+2*n]+M[A+2*m,A+2*n+1]+M[A+2*m+1,A+2*n+1]>0:
        return False

    if n==0:
        tmp_w = True
        tmp_e = (M[A+2*m,A+2*(n+1)]+M[A+2*m+1,A+2*(n+1)])==0
    elif n==k:
        tmp_w = (M[A+2*m,A+2*n-1]+M[A+2*m+1,A+2*n-1])==0
        tmp_e = True
    else:
        tmp_w = (M[A+2*m,A+2*n-1]+M[A+2*m+1,A+2*n-1])==0
        tmp_e = (M[A+2*m,A+2*(n+1)]+M[A+2*m+1,A+2*(n+1)])==0

    if m==0:
        tmp_n = True
        tmp_s = (M[A+2*(m+1),A+2*n]+M[A+2*(m+1),A+2*n+1])==0
    elif m==k:
        tmp_n = (M[A+2*m-1,A+2*n]+M[A+2*m-1,A+2*n+1])==0
        tmp_s = True
    else:
        tmp_n = (M[A+2*m-1,A+2*n]+M[A+2*m-1,A+2*n+1])==0
        tmp_s = (M[A+2*(m+1),A+2*n]+M[A+2*(m+1),A+2*n+1])==0

    return tmp_s and tmp_e and tmp_w and tmp_n

@njit("(ListType(Array(float64, 2, 'C')),)")
def shuffling(C):
    N = 2*len(C)
    M = zeros((N,N),dtype=int8)
    #print(N)

    # Initiate
    A = N//2-1
    w,x,y,z = C[0][0,0],C[0][0,1],C[0][1,0],C[0][1,1]

    tmp = (w*z+x*y)


    prob = (w*z)/tmp
    if random()<prob:
        M[A,A],M[A+1,A+1]= 1,1
        M[A,A+1],M[A+1,A]= 0,0
    else:
        M[A,A],M[A+1,A+1]= 0,0
        M[A,A+1],M[A+1,A]= 1,1

    for k in range(1,N//2):
        A = N//2-1-k   
        # destruction + flip
        for m in range(k+1):
            for n in range(k+1):
                tmp = M[A+2*m,A+2*n]+M[A+2*m+1,A+2*n]+M[A+2*m,A+2*n+1]+M[A+2*m+1,A+2*n+1]
                if tmp>1:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n+1]=0,0,0,0
                elif tmp == 1:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]=M[A+2*m+1,A+2*n+1],M[A+2*m,A+2*n]
                    M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]=M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1]

        for m in range(k+1):
            for n in range(k+1):
                if no_neighbor(M,k,m,n):
                    w,x,y,z = C[k][2*m,2*n],C[k][2*m,2*n+1],C[k][2*m+1,2*n],C[k][2*m+1,2*n+1]
                    prob = (w*z)/(w*z+x*y)
                    if random()<prob:
                        M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 1,1
                        M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 0,0
                    else:
                        M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 0,0
                        M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 1,1
    return M


@njit("(Array(int8, 2, 'C'),int64,Array(float64, 2, 'C'),int64)")
def shuffling_helper(M,A,C,k):
    # destruction + flip
    for m in range(k+1):
        for n in range(k+1):
            tmp = M[A+2*m,A+2*n]+M[A+2*m+1,A+2*n]+M[A+2*m,A+2*n+1]+M[A+2*m+1,A+2*n+1]
            if tmp>1:
                M[A+2*m,A+2*n],M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n+1]=0,0,0,0
            elif tmp == 1:
                M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]=M[A+2*m+1,A+2*n+1],M[A+2*m,A+2*n]
                M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]=M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1]

    for m in range(k+1):
        for n in range(k+1):
            if no_neighbor(M,k,m,n):
                w,x,y,z = C[2*m,2*n],C[2*m,2*n+1],C[2*m+1,2*n],C[2*m+1,2*n+1]
                prob = (w*z)/(w*z+x*y)
                if random()<prob:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 1,1
                    M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 0,0
                else:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 0,0
                    M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 1,1


def shuffling_hd(N):
    M = zeros((N,N),dtype=int8)
    
    C = load('CTower/0.npy')
    # Initiate
    A = N//2-1
    w,x,y,z = C[0,0],C[0,1],C[1,0],C[1,1]

    tmp = (w*z+x*y)


    prob = (w*z)/tmp
    if random()<prob:
        M[A,A],M[A+1,A+1]= 1,1
        M[A,A+1],M[A+1,A]= 0,0
    else:
        M[A,A],M[A+1,A+1]= 0,0
        M[A,A+1],M[A+1,A]= 1,1
    
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    for k in tqdm(range(1,N//2),desc="Shuffling",bar_format=bar_format):
        C = load('CTower/'+str(k)+'.npy')
        A = N//2-1-k   
        
        shuffling_helper(M,A,C,k)
    return M


@njit("(Array(float64, 2, 'C', False, aligned=True),)")
def shuffling_memory(C0):
    C = C0.copy()
    N = C.shape[0]
    M = zeros((N,N),dtype=int8)

    # Initiate
    A = N//2-1
    w,x,y,z = C[A,A],C[A,A+1],C[A+1,A],C[A+1,A+1]

    tmp = (w*z+x*y)


    prob = (w*z)/tmp
    if random()<prob:
        M[A,A],M[A+1,A+1]= 1,1
        M[A,A+1],M[A+1,A]= 0,0
    else:
        M[A,A],M[A+1,A+1]= 0,0
        M[A,A+1],M[A+1,A]= 1,1

    for k in range(1,N//2):
        A = N//2-1-k   

        # destruction + flip (+ reconstruct)
        for m in range(k+1):
            for n in range(k+1):
                # reconstruct
                w,x,y,z = C[A+2*m,A+2*n],C[A+2*m+1,A+2*n],C[A+2*m,A+2*n+1],C[A+2*m+1,A+2*n+1]
                d = w*z+x*y
                C[A+2*m,A+2*n],C[A+2*m+1,A+2*n+1] = C[A+2*m+1,A+2*n+1]/d, C[A+2*m,A+2*n]/d
                C[A+2*m,A+2*n+1],C[A+2*m+1,A+2*n] = C[A+2*m+1,A+2*n]/d, C[A+2*m,A+2*n+1]/d

                # destruction + flip
                tmp = M[A+2*m,A+2*n]+M[A+2*m+1,A+2*n]+M[A+2*m,A+2*n+1]+M[A+2*m+1,A+2*n+1]
                if tmp>1:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n+1]=0,0,0,0
                elif tmp == 1:
                    M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]=M[A+2*m+1,A+2*n+1],M[A+2*m,A+2*n]
                    M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]=M[A+2*m+1,A+2*n],M[A+2*m,A+2*n+1]

        for m in range(k+1):
            for n in range(k+1):
                if no_neighbor(M,k,m,n):
                    w,x,y,z = C[A+2*m,A+2*n],C[A+2*m,A+2*n+1],C[A+2*m+1,A+2*n],C[A+2*m+1,A+2*n+1]
                    prob = (w*z)/(w*z+x*y)
                    if random()<prob:
                        M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 1,1
                        M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 0,0
                    else:
                        M[A+2*m,A+2*n],M[A+2*m+1,A+2*n+1]= 0,0
                        M[A+2*m,A+2*n+1],M[A+2*m+1,A+2*n]= 1,1
    return M
