from numba import jit
import numpy as np

@jit()
def counting_one(w1,w2,w3,w4):
    I14 = w1.imag + w4.imag
    I23 = w2.imag + w3.imag
    if I14>I23:
        return w2.real*w3.real + I23*1j
    elif I14<I23:
        return w1.real*w4.real + I14*1j
    else:
        return w1.real*w4.real + w2.real*w3.real + I14*1j

@jit()
def logcounting(Wi):
    l = Wi.shape[0]//2
    logZni = 0
    for i1 in range(l):
        for i2 in range(l):
            incr = counting_one(Wi[2*i1,2*i2],Wi[2*i1+1,2*i2],Wi[2*i1,2*i2+1],Wi[2*i1+1,2*i2+1])
            logZni = logZni.real + np.log(incr.real) + (logZni.imag+incr.imag)*1j
    return logZni

@jit("(List(Array(complex128, 2, 'C'), True),)")
def log_partition_function(C):
    logZn = 0
    for i in range(len(C)):
        Wi = C[i]
        logci = logcounting(Wi)
        logZn = logZn.real + logci.real + (logZn.imag+logci.imag)*1j
    return logZn