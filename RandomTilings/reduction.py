from numba import jit
import numpy as np

@jit()
def reduction(W):
    N = W.shape[0]
    W_red = np.zeros((N-2,N-2))+0j

    for x in range(1,N-1):
        for y in range(1,N-1):
            if x%2 == 1 and y%2 == 1:
                W_red[N-2-y,x-1] = red_one(W[N-y-1,x-1],W[N-y,x-1],W[N-y-1,x],W[N-y,x],3)
            elif x%2 == 0 and y%2 == 1:
                W_red[N-2-y,x-1] = red_one(W[N-y-1,x],W[N-y,x],W[N-y-1,x+1],W[N-y,x+1],1)
            elif x%2 == 1 and y%2 == 0:
                W_red[N-2-y,x-1] = red_one(W[N-y-2,x-1],W[N-y-1,x-1],W[N-y-2,x],W[N-y-1,x],4)
            else:
                W_red[N-2-y,x-1] = red_one(W[N-y-2,x],W[N-y-1,x],W[N-y-2,x+1],W[N-y-1,x+1],2)
    return W_red

@jit()
def red_one(w1,w2,w3,w4,ind):
    I14 = w1.imag + w4.imag
    I23 = w2.imag + w3.imag
    if I14 > I23:
        D = 1/(w2.real*w3.real) - 1j*I23
    elif I14 < I23:
        D = 1/(w1.real*w4.real) - 1j*I14
    else:
        D = 1/(w1.real*w4.real + w2.real*w3.real) - 1j*I14

    if ind == 1:
        return w4.real*D.real + 1j*(w4.imag+D.imag)
    elif ind == 2:
        return w3.real*D.real + 1j*(w3.imag+D.imag)
    elif ind == 3:
        return w2.real*D.real + 1j*(w2.imag+D.imag)
    else:
        return w1.real*D.real + 1j*(w1.imag+D.imag)