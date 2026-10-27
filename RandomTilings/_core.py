from numpy import zeros,int8,exp,log,inf,array,empty,vstack,hstack
from numpy.random import random
from numba import njit,prange
from numba.typed import List

@njit("(float64, float64, float64, float64, int32, int32, int32, int32)",cache=True,inline="always")
def update(w,x,y,z,ew,ex,ey,ez):
    overflow = False
    if w==0:
        w        = 1
        ew      += 1
        overflow = True
    elif 1/w==0:
        w        = 1
        ew      -= 1
        overflow = True
    if x==0:
        x        = 1
        ex      += 1
        overflow = True
    elif 1/x==0:
        x        = 1
        ex      -= 1
        overflow = True
    if y==0:
        y        = 1
        ey      += 1
        overflow = True
    elif 1/y==0:
        y        = 1
        ey      -= 1
        overflow = True

    if z==0:
        z        = 1
        ez      += 1
        overflow = True
    elif 1/z==0:
        z        = 1
        ez      -= 1
        overflow = True

    Ixy = ex+ey
    Iwz = ew+ez

    if Ixy > Iwz:
        tmp = w*z
        if tmp == 0:
           return 1/w,1,1,1/z,-ew,ey-Iwz-1,ex-Iwz-1,-ez,True
        elif 1/tmp ==0:
            return 1/w,1,1,1/z,-ew,ey-Iwz+1,ex-Iwz+1,-ez,True
        else:
            return 1/w,y/tmp,x/tmp,1/z,-ew,ey-Iwz,ex-Iwz,-ez,overflow
    elif Ixy < Iwz:
        tmp = x*y
        if tmp == 0:
            return 1,1/x,1/y,1,ez-Ixy-1,-ex,-ey,ew-Ixy-1,True
        elif 1/tmp ==0:
            return 1,1/x,1/y,1,ez-Ixy+1,-ex,-ey,ew-Ixy+1,True
        else:
            return z/tmp,1/x,1/y,w/tmp,ez-Ixy,-ex,-ey,ew-Ixy,overflow
    else:
        tmp = z*w+x*y
        if tmp == 0:
            return 1, 1, 1, 1, ez-Ixy-1,ey-Ixy-1,ex-Ixy-1,ew-Ixy-1,True
        elif 1/tmp == 0:
            return 1, 1, 1, 1, ez-Ixy+1,ey-Ixy+1,ex-Ixy+1,ew-Ixy+1,True
        else:
            return z/tmp, y/tmp, x/tmp, w/tmp, ez-Ixy,ey-Ixy,ex-Ixy,ew-Ixy,overflow

@njit("(float64, float64, float64, float64, int32, int32, int32, int32)",cache=True,inline="always")
def comp_prop(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Ixy>Iwz:
        return 1.,False
    elif Ixy < Iwz:
        return 0.,False
    tmp = (w*z+x*y)
    if tmp == 0 or 1/tmp == 0:
        return 1/2, True
    else:
        return w*z/(w*z+x*y),False

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def count(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Iwz > Ixy:
        return log(x*y) + Ixy*1j
    elif Iwz < Ixy:
        return log(w*z) + Iwz*1j
    else:
        return log(w*z + x*y) + Iwz*1j

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",cache=True,parallel=True)
def reduce_weight(W,E):
    K = W.shape[0]//2
    C = W.copy()+(W==0)
    E+= 1*(W == 0)
    overflow = 0
    log_zn = 0
    for k in range(K-1):
        for m in prange(K-k):
            for n in prange(K-k):
                I00,I01,I10,I11 = k+2*m,k+2*n,k+2*m+1,k+2*n+1
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                log_zn += count(w,x,y,z,ew,ex,ey,ez)
                w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez
                overflow+= of
    k = K-1
    I00,I01,I10,I11 = k,k,k+1,k+1
    w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
    ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
    log_zn += count(w,x,y,z,ew,ex,ey,ez)
    return C,E,log_zn,overflow>0

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",
      cache=True,parallel=True)
def shuffling(C0,E0):
    C = C0.copy()
    E = E0.copy()
    N = C.shape[0]
    M = zeros((N,N),dtype=int8)
    overflow = False

    # initiate
    A = N//2-1
    w,x,y,z     = C[A,A],C[A,A+1],C[A+1,A],C[A+1,A+1]
    ew,ex,ey,ez = E[A,A],E[A,A+1],E[A+1,A],E[A+1,A+1]
    prob,of = comp_prop(w,x,y,z,ew,ex,ey,ez)
    overflow = overflow or of
    if random()<prob:
        M[A,A],M[A+1,A+1]= 1,1
        M[A,A+1],M[A+1,A]= 0,0
    else:
        M[A,A],M[A+1,A+1]= 0,0
        M[A,A+1],M[A+1,A]= 1,1

    for k in range(1,N//2):
        A = N//2-1-k   

        # destruction + flip (+ reconstruction)
        for m in range(k+1):
            for n in range(k+1):
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                # reconstruction
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                
                w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)
                overflow = overflow or of
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez

                # destruction + flip
                tmp = M[I00,I01]+M[I10,I01]+M[I00,I11]+M[I10,I11]
                if tmp>1:
                    M[I00,I01],M[I10,I01],M[I00,I11],M[I10,I11]=0,0,0,0
                elif tmp == 1:
                    M[I00,I01],M[I10,I11]=M[I10,I11],M[I00,I01]
                    M[I00,I11],M[I10,I01]=M[I10,I01],M[I00,I11]

        for m in range(k+1):
            for n in range(k+1):
                # start checking for neighbor
                A = N//2-1-k  
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                
                if M[I00,I01]+M[I10,I01]+M[I00,I11]+M[I10,I11]>0:
                    no_neighbor = False
                else:
                    if n==0:
                        tmp_w = True
                        tmp_e = (M[I00,I11+1]+M[I10,I11+1])==0
                    elif n==k:
                        tmp_w = (M[I00,I01-1]+M[I10,I01-1])==0
                        tmp_e = True
                    else:
                        tmp_w = (M[I00,I01-1]+M[I10,I01-1])==0
                        tmp_e = (M[I00,I11+1]+M[I10,I11+1])==0

                    if m==0:
                        tmp_n = True
                        tmp_s = (M[I10+1,I01]+M[I10+1,I11])==0
                    elif m==k:
                        tmp_n = (M[I00-1,I01]+M[I00-1,I11])==0
                        tmp_s = True
                    else:
                        tmp_n = (M[I00-1,I01]+M[I00-1,I11])==0
                        tmp_s = (M[I00+1,I01]+M[I00+1,I11])==0
                    no_neighbor = tmp_s and tmp_e and tmp_w and tmp_n
                # end checking for neighbor

                if no_neighbor:
                    w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                    ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                    prob,of  = comp_prop(w,x,y,z,ew,ex,ey,ez)
                    overflow = overflow or of

                    if random()<prob:
                        M[I00,I01],M[I10,I11]= 1,1
                        M[I00,I11],M[I10,I01]= 0,0
                    else:
                        M[I00,I01],M[I10,I11]= 0,0
                        M[I00,I11],M[I10,I01]= 1,1
    return M,overflow

###############
# Log variant #
###############

@njit("(float64,float64)",cache=True,inline="always")
def logaddexp(a,b):
    return max(a,b)+log(1+exp(-abs(a-b)))

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def update_log(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Ixy>Iwz:
        d           = w+z 
        w,x,y,z     = -w,y-d,x-d,-z
        ew,ex,ey,ez = -ew,ey-Iwz,ex-Iwz,-ez
    elif Ixy<Iwz:
        d           = x+y
        w,x,y,z     = z-d,-x,-y,w-d
        ew,ex,ey,ez = ez-Ixy,-ex,-ey,ew-Ixy
    else:
        d           = logaddexp(w+z,x+y)
        w,x,y,z     = z-d, y-d, x-d, w-d
        ew,ex,ey,ez = ez-Ixy,ey-Ixy,ex-Ixy,ew-Ixy
    return w,x,y,z,ew,ex,ey,ez

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def comp_prop_log(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Ixy>Iwz:
        return 0
    elif Ixy < Iwz:
        return -inf
    tmp = logaddexp(w+z,x+y)
    if tmp == -inf:
        return -log(2)
    else:
        return w+z-tmp

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def count_log(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Iwz > Ixy:
        return x+y + Ixy*1j
    elif Iwz < Ixy:
        return w+z + Iwz*1j
    else:
        return logaddexp(w+z,x+y) + Iwz*1j

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",cache=True,parallel=True)
def reduce_weight_log(W,E):
    K = W.shape[0]//2
    C = log(W.copy()+(W==0))
    E+= 1*(W == 0)
    log_zn = 0
    for k in range(K-1):
        for m in prange(K-k):
            for n in prange(K-k):
                I00,I01,I10,I11 = k+2*m,k+2*n,k+2*m+1,k+2*n+1
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                log_zn += count_log(w,x,y,z,ew,ex,ey,ez)
                w,x,y,z,ew,ex,ey,ez = update_log(w,x,y,z,ew,ex,ey,ez)
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez
    k = K-1
    I00,I01,I10,I11 = k,k,k+1,k+1
    w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
    ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
    log_zn += count_log(w,x,y,z,ew,ex,ey,ez)
    return C,E,log_zn

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",
      cache=True,parallel=True)
def shuffling_log(C0,E0):
    C = C0.copy()
    E = E0.copy()
    N = C.shape[0]
    M = zeros((N,N),dtype=int8)

    # initiate
    A = N//2-1
    w,x,y,z     = C[A,A],C[A,A+1],C[A+1,A],C[A+1,A+1]
    ew,ex,ey,ez = E[A,A],E[A,A+1],E[A+1,A],E[A+1,A+1]
    prob = comp_prop_log(w,x,y,z,ew,ex,ey,ez)
    if log(random())<prob:
        M[A,A],M[A+1,A+1]= 1,1
        M[A,A+1],M[A+1,A]= 0,0
    else:
        M[A,A],M[A+1,A+1]= 0,0
        M[A,A+1],M[A+1,A]= 1,1

    for k in range(1,N//2):
        A = N//2-1-k   

        # destruction + flip (+ reconstruction)
        for m in range(k+1):
            for n in range(k+1):
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                # reconstruction
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                
                w,x,y,z,ew,ex,ey,ez = update_log(w,x,y,z,ew,ex,ey,ez)
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez

                # destruction + flip
                tmp = M[I00,I01]+M[I10,I01]+M[I00,I11]+M[I10,I11]
                if tmp>1:
                    M[I00,I01],M[I10,I01],M[I00,I11],M[I10,I11]=0,0,0,0
                elif tmp == 1:
                    M[I00,I01],M[I10,I11]=M[I10,I11],M[I00,I01]
                    M[I00,I11],M[I10,I01]=M[I10,I01],M[I00,I11]

        for m in range(k+1):
            for n in range(k+1):
                # start checking for neighbor
                A = N//2-1-k  
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                
                if M[I00,I01]+M[I10,I01]+M[I00,I11]+M[I10,I11]>0:
                    no_neighbor = False
                else:
                    if n==0:
                        tmp_w = True
                        tmp_e = (M[I00,I11+1]+M[I10,I11+1])==0
                    elif n==k:
                        tmp_w = (M[I00,I01-1]+M[I10,I01-1])==0
                        tmp_e = True
                    else:
                        tmp_w = (M[I00,I01-1]+M[I10,I01-1])==0
                        tmp_e = (M[I00,I11+1]+M[I10,I11+1])==0

                    if m==0:
                        tmp_n = True
                        tmp_s = (M[I10+1,I01]+M[I10+1,I11])==0
                    elif m==k:
                        tmp_n = (M[I00-1,I01]+M[I00-1,I11])==0
                        tmp_s = True
                    else:
                        tmp_n = (M[I00-1,I01]+M[I00-1,I11])==0
                        tmp_s = (M[I00+1,I01]+M[I00+1,I11])==0
                    no_neighbor = tmp_s and tmp_e and tmp_w and tmp_n
                # end checking for neighbor

                if no_neighbor:
                    w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                    ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                    prob  = comp_prop_log(w,x,y,z,ew,ex,ey,ez)

                    if log(random())<prob:
                        M[I00,I01],M[I10,I11]= 1,1
                        M[I00,I11],M[I10,I01]= 0,0
                    else:
                        M[I00,I01],M[I10,I11]= 0,0
                        M[I00,I11],M[I10,I01]= 1,1
    return M

####################
# periodic variant #
####################

@njit("(float64,float64)",cache=True,inline="always")
def logaddexp(a,b):
    return max(a,b)+log(1+exp(-abs(a-b)))

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",
      cache=True,inline='always')
def update_weight(W,E):
    M,N = W.shape
    M,N = M//2,N//2

    W1 = W.copy()  # FIX: update() already handles numerical zeros
    E1 = E.copy()  # FIX
    overflow = 0

    for m in range(M):
        for n in range(N):
            w,x,y,z     = W1[2*m,2*n],W1[2*m,2*n+1],W1[2*m+1,2*n],W1[2*m+1,2*n+1]
            ew,ex,ey,ez = E1[2*m,2*n],E1[2*m,2*n+1],E1[2*m+1,2*n],E1[2*m+1,2*n+1]

            w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)

            W1[2*m,2*n],W1[2*m,2*n+1],W1[2*m+1,2*n],W1[2*m+1,2*n+1] = w,x,y,z
            E1[2*m,2*n],E1[2*m,2*n+1],E1[2*m+1,2*n],E1[2*m+1,2*n+1] = ew,ex,ey,ez

            overflow += of

    I1 = array([k for k in range(1,2*N)] + [0])
    I2 = array([-1]+[k for k in range(2*M-1)])

    W1 = W1[:,I1]
    W1 = W1[I2]
    E1 = E1[:,I1]
    E1 = E1[I2]
    return W1,E1,overflow

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",
      cache=True,inline='always')
def update_weight_log(W,E):
    M,N = W.shape
    M,N = M//2,N//2

    W1 = W.copy()  # FIX: update() already handles numerical zeros
    E1 = E.copy()  # FIX

    for m in range(M):
        for n in range(N):
            w,x,y,z     = W1[2*m,2*n],W1[2*m,2*n+1],W1[2*m+1,2*n],W1[2*m+1,2*n+1]
            ew,ex,ey,ez = E1[2*m,2*n],E1[2*m,2*n+1],E1[2*m+1,2*n],E1[2*m+1,2*n+1]

            w,x,y,z,ew,ex,ey,ez = update_log(w,x,y,z,ew,ex,ey,ez)

            W1[2*m,2*n],W1[2*m,2*n+1],W1[2*m+1,2*n],W1[2*m+1,2*n+1] = w,x,y,z
            E1[2*m,2*n],E1[2*m,2*n+1],E1[2*m+1,2*n],E1[2*m+1,2*n+1] = ew,ex,ey,ez

    I1 = array([k for k in range(1,2*N)] + [0])
    I2 = array([-1]+[k for k in range(2*M-1)])

    W1 = W1[:,I1]
    W1 = W1[I2]
    E1 = E1[:,I1]
    E1 = E1[I2]
    return W1,E1

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))"
      ,cache=True,inline='always')
def cProp(W,E):
    L1,L2 = W.shape
    P = empty((L1//2,L2//2))
    overflow = 0
    for m in range(L1//2):
        for n in range(L2//2):
            w,x,y,z     = W[2*m,2*n],W[2*m,2*n+1],W[2*m+1,2*n],W[2*m+1,2*n+1]
            ew,ex,ey,ez = E[2*m,2*n],E[2*m,2*n+1],E[2*m+1,2*n],E[2*m+1,2*n+1]
            P[m,n],of   = comp_prop(w,x,y,z,ew,ex,ey,ez)
            overflow   += of

    return P, overflow

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))"
      ,cache=True,inline='always')
def cProp_log(W,E):
    L1,L2 = W.shape
    P = empty((L1//2,L2//2))
    for m in range(L1//2):
        for n in range(L2//2):
            w,x,y,z     = W[2*m,2*n],W[2*m,2*n+1],W[2*m+1,2*n],W[2*m+1,2*n+1]
            ew,ex,ey,ez = E[2*m,2*n],E[2*m,2*n+1],E[2*m+1,2*n],E[2*m+1,2*n+1]
            P[m,n]      = exp(comp_prop_log(w,x,y,z,ew,ex,ey,ez))
    return P

@njit("(int64,Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True),bool)",
      cache=True)
def PropTower(n,W,E,log=False):
    L1,L2 = W.shape
    if L1%2==1:
        W = vstack((W,W))
        E = vstack((E,E))
    if L2%2==1:
        W = hstack((W,W))
        E = hstack((E,E))

    L1,L2 = W.shape
    if log:
        P  = cProp_log(W,E)
        of = False
    else:
        P,of  = cProp(W,E)
    PT = List([P])
    overflow = of
    for k in range(n-1):
        if log:
            W,E = update_weight_log(W,E)
            P   = cProp_log(W,E)
        else:
            W,E,of1 = update_weight(W,E)
            P,of2 = cProp(W,E)
            overflow += of1+of2

        PT.append(P)
    return List(PT[::-1]), overflow  # FIX: reverse probability tower

@njit("(int64, int64, int64, int64, int64)",inline='always',cache=True)
def idx_correction(Size,WShape1,WShape2,m,n):
    m = WShape1 - (Size-m-1) % WShape1 - 1
    n = n % WShape2
    return m,n

@njit("(ListType(Array(float64, 2, 'C', False, aligned=True)),)",cache=True)
def shuffling_periodic(PT):
    K = len(PT)
    N = 2*K
    M = zeros((N,N),dtype=int8)

    L1,L2 = PT[0].shape

    # Initiate
    A = K-1
    prob = PT[0][L1-1,0]

    if random()<prob:
        M[A,A],M[A+1,A+1] = 1,1
    else:
        M[A,A+1],M[A+1,A] = 1,1

    for k in range(1,K):
        A = K-1-k
        P = PT[k]

        for m in range(k+1):
            i = A+2*m
            r = (m-k-1)%L1
            s = 0

            for n in range(k+1):
                j = A+2*n

                a = M[i,j]
                b = M[i,j+1]
                c = M[i+1,j]
                d = M[i+1,j+1]

                tmp = a+b+c+d

                if tmp == 1:
                    # Flip
                    M[i,j],M[i,j+1] = d,c
                    M[i+1,j],M[i+1,j+1] = b,a

                else:
                    # Check only previously processed neighbors
                    free_w = (n==0 or
                              (M[i,j-1]==0 and M[i+1,j-1]==0))

                    free_n = (m==0 or
                              (M[i-1,j]==0 and M[i-1,j+1]==0))

                    if free_w and free_n:
                        prob = P[r,s]

                        if random()<prob:
                            M[i,j],M[i,j+1] = 1,0
                            M[i+1,j],M[i+1,j+1] = 0,1
                        else:
                            M[i,j],M[i,j+1] = 0,1
                            M[i+1,j],M[i+1,j+1] = 1,0

                    elif tmp>1:
                        # Destruction without reconstruction
                        M[i,j],M[i,j+1] = 0,0
                        M[i+1,j],M[i+1,j+1] = 0,0

                # Cyclic probability-column index
                s += 1
                if s==L2:
                    s = 0

    return M