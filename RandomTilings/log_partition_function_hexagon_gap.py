from RandomTilings.weight_hexagon import weight_hexagon
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.log_partition_function import log_partition_function
import numpy as np

def log_partition_function_hexagon_gap(n,w,gap,a=1,b=1,c=1):
    A = int(np.round(a*n))
    B = int(np.round(b*n))
    C = int(np.round(c*n))
    N = 2*(A+B+C-1)
    end = N

    w = w.astype(float)
    W = weight_hexagon(n,w,a,b,c)
    C = algorithm_reduction_weight(W)
    logZnDen = log_partition_function(C)

    size_gap = gap.shape[0]
    for i1 in range(size_gap):
        X = int(2*gap[i1][0])
        for Y in range(int(2*gap[i1][1]),int(2*(gap[i1][2]+1)),2):
            xUp = X-1
            yUp = Y-1
            if 1 <= N-(yUp-1) and N-(yUp-1) <= N and 1 <= xUp and xUp <= N:
                W[end-yUp,xUp-1] = 0
            xHorizontal = X-1
            yHorizontal = Y
            if 1 <= N-(yHorizontal-1) and N-(yHorizontal-1) <= N and 1 <= xHorizontal and xHorizontal <= N:
                W[end-yHorizontal,xHorizontal-1] = 0
    
    C = algorithm_reduction_weight(W)
    logZnNum = log_partition_function(C)
    logPn = logZnNum - logZnDen
    return [logPn, logZnNum, logZnDen]