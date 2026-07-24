from ._core import update,update_log,logaddexp
from numpy import log
from numba import njit

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def counting_one(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Iwz > Ixy:
        return x*y + Ixy*1j
    elif Iwz < Ixy:
        return w*z + Iwz*1j
    else:
        return w*z + x*y + Iwz*1j

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True),int32,int32)",cache=True,inline="always")
def log_counting(C,E,A,i):
    log_zni = 0
    for m in range(i+1):
        for n in range(i+1):
            I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
            w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
            ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
            incr = counting_one(w,x,y,z,ew,ex,ey,ez)
            log_zni = log_zni.real + log(incr.real) + (log_zni.imag + incr.imag)*1j
    return log_zni

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",cache=True)
def log_partition_function(C0,E0):
    C = C0.copy()
    E = E0.copy()
    N = C.shape[0]
    overflow = False

    log_ci = log_counting(C,E,N//2-1,0)
    log_zn = log_ci.real + log_ci.imag*1j

    for i in range(1,N//2):
        A = N//2-1-i

        # reconstruction
        for m in range(i+1):
            for n in range(i+1):
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]

                w,x,y,z,ew,ex,ey,ez,of = update(w,x,y,z,ew,ex,ey,ez)
                overflow = overflow or of
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez

        # compute the contribution of the current C
        log_ci = log_counting(C,E,A,i)
        log_zn = log_zn.real + log_ci.real + (log_zn.imag + log_ci.imag)*1j
    return log_zn, overflow

###############
# Log variant #
###############

@njit("(float64,float64,float64,float64,int32,int32,int32,int32)",cache=True,inline="always")
def counting_one_log(w,x,y,z,ew,ex,ey,ez):
    Ixy = ex+ey
    Iwz = ew+ez
    if Iwz > Ixy:
        return x+y + Ixy*1j
    elif Iwz < Ixy:
        return w+z + Iwz*1j
    else:
        return logaddexp(w+z,x+y) + Iwz*1j

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True),int32,int32)",cache=True,inline="always")
def log_counting_log(C,E,A,i):
    log_zni = 0
    for m in range(i+1):
        for n in range(i+1):
            I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
            w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
            ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]
            incr = counting_one_log(w,x,y,z,ew,ex,ey,ez)
            log_zni = log_zni.real + incr.real + (log_zni.imag + incr.imag)*1j
    return log_zni

@njit("(Array(float64, 2, 'C', False, aligned=True),Array(int32, 2, 'C', False, aligned=True))",cache=True)
def log_partition_function_log(C0,E0):
    C = C0.copy()
    E = E0.copy()
    N = C.shape[0]

    log_ci = log_counting_log(C,E,N//2-1,0)
    log_zn = log_ci.real + log_ci.imag*1j

    for i in range(1,N//2):
        A = N//2-1-i

        # reconstruction
        for m in range(i+1):
            for n in range(i+1):
                I00,I01,I10,I11 = A+2*m,A+2*n,A+2*m+1,A+2*n+1
                w,x,y,z     = C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11]
                ew,ex,ey,ez = E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11]

                w,x,y,z,ew,ex,ey,ez = update_log(w,x,y,z,ew,ex,ey,ez)
                C[I00,I01],C[I00,I11],C[I10,I01],C[I10,I11] = w,x,y,z
                E[I00,I01],E[I00,I11],E[I10,I01],E[I10,I11] = ew,ex,ey,ez

        # compute the contribution of the current C
        log_ci = log_counting_log(C,E,A,i)
        log_zn = log_zn.real + log_ci.real + (log_zn.imag + log_ci.imag)*1j
    return log_zn