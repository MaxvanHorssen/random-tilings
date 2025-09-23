from numba import jit

@jit("(complex128, complex128, complex128, complex128)")
def probability_matching_2times2(w1,w2,w3,w4):
    I14 = w1.imag + w4.imag
    I23 = w2.imag + w3.imag  
    if I14 > I23:
        return 0
    elif I14 < I23:
        return 1
    else:
        return w1.real*w4.real/(w1.real*w4.real + w2.real*w3.real)