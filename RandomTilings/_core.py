from numpy import zeros,load,save,int8,int32
from numpy.random import random
from numba import njit,prange
from tqdm import tqdm
import os
@njit("(Array(int8, 2, 'C', False, aligned=True), int64, int64, int64)",cache=True,inline = "always")
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


    if Ixy>Iwz:
        tmp = w*z
        if tmp == 0:
           return 1/w,1,1,1/z,-ew,ey-Iwz-1,ex-Iwz-1,-ez,True
        elif 1/tmp ==0:
            return 1/w,1,1,1/z,-ew,ey-Iwz+1,ex-Iwz+1,-ez,True
        else:
            return 1/w,y/tmp,x/tmp,1/z,-ew,ey-Iwz,ex-Iwz,-ez,overflow
    elif Ixy<Iwz:
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

@njit("(Array(float64, 2, 'C', False, aligned=True),)",cache=True,parallel = True)
def reduce_weight(W):
    K = W.shape[0] // 2
    C = W.copy()+ (W==0)
    E = zeros((2*K,2*K),dtype=int32)
    E+= 1*(W == 0) #.astype(int)
    overflow = False
    for k in range(K-1):
        for m in prange(K-k):
            for n in prange(K-k):
                I00,I01,I10,I11 = k+2*m,k+2*n,k+2*m+1,k+2*n+1
                w ,x ,y ,z  = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                
                w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez
                overflow = overflow or of
    return C,E,overflow

# @njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",cache=True)
# def shuffling(C0,E0):
#     C = C0.copy()
#     E = E0.copy()
#     N = C.shape[0]
#     M = zeros((N,N),dtype=int8)
#     overflow = False

#     # Initiate
#     A = N//2-1
#     w,x,y,z     = C[A,A],C[A,A+1],C[A+1,A],C[A+1,A+1]
#     ew,ex,ey,ez = E[A,A],E[A,A+1],E[A+1,A],E[A+1,A+1]
#     prob,of = comp_prop(w,x,y,z,ew,ex,ey,ez)
#     overflow = overflow or of
#     if random()<prob:
#         M[A,A],M[A+1,A+1]= 1,1
#         M[A,A+1],M[A+1,A]= 0,0
#     else:
#         M[A,A],M[A+1,A+1]= 0,0
#         M[A,A+1],M[A+1,A]= 1,1

    
#     for k in range(1,N//2):
#         A = N//2-1-k   

#         # destruction + flip (+ reconstruct)
#         for m in range(k+1):
#             for n in range(k+1):
#                 I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
#                 # reconstruct
#                 w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
#                 ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
                
#                 w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)
#                 overflow = overflow or of
#                 C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
#                 E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez

#                 # destruction + flip
#                 tmp = M[I00,I01]+M[I10,I01]+M[I00,I11]+M[I10,I11]
#                 if tmp>1:
#                     M[I00,I01],M[I10,I01],M[I00,I11],M[I10,I11]=0,0,0,0
#                 elif tmp == 1:
#                     M[I00,I01],M[I10,I11]=M[I10,I11],M[I00,I01]
#                     M[I00,I11],M[I10,I01]=M[I10,I01],M[I00,I11]

#         for m in range(k+1):
#             for n in range(k+1):
#                 if no_neighbor(M,k,m,n):
#                     I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
#                     w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
#                     ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
#                     prob,of  = comp_prop(w,x,y,z,ew,ex,ey,ez)
#                     overflow = overflow or of

#                     if random()<prob:
#                         M[I00,I01],M[I10,I11]= 1,1
#                         M[I00,I11],M[I10,I01]= 0,0
#                     else:
#                         M[I00,I01],M[I10,I11]= 0,0
#                         M[I00,I11],M[I10,I01]= 1,1
#     return M,overflow



@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",
      cache=True,parallel = True)
def shuffling(C0,E0):
    C = C0.copy()
    E = E0.copy()
    N = C.shape[0]
    M = zeros((N,N),dtype=int8)
    overflow = False

    # Initiate
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

        # destruction + flip (+ reconstruct)
        for m in range(k+1):
            for n in range(k+1):
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                # reconstruct
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
                ####################
                # Check for Neighboor
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

                # end checking for neighboor

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



######################################################################################
# Hard Drive Mode
######################################################################################

@njit("(Array(int8, 2, 'C'),int64,Array(float64, 2, 'C'),int64)",cache=True)
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



@njit("(float64, float64, float64, float64)",cache=True)
def red_cases_tmp(w,x,y,z): 
    if w == 0:
        tmp_w = 0
    elif z == 0:
        tmp_w = w/(x*y)
    elif x*y == 0:
        tmp_w = 1/z
    elif 1/w==0: 
        tmp_w = 1/z
    elif 1/(x*y)==0 or 1/z ==0:
        tmp_w = 0
    else:
        tmp_w = w/(w*z+x*y)
    return tmp_w





@njit("(float64, float64, float64, float64)",cache=True)
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


@njit(["(Array(float64, 2, 'C', False, aligned=True),)","(Array(float64, 2, 'A', False, aligned=True),)"],cache=True)
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

def clear_CTower():
    os.makedirs("CTower", exist_ok=True)
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


