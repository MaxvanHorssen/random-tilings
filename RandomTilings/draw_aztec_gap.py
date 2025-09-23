from RandomTilings.weight_aztec import weight_aztec
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.log_partition_function import log_partition_function
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_dominos import draw_dominos

def draw_aztec_gap(n,w,gap,edge=0,dpi=200,rotated=True,show_figure=True):
    N = 2*n
    end = N

    w = w.astype(float)
    W = weight_aztec(n,w)
    C = algorithm_reduction_weight(W)
    logZnDen = log_partition_function(C)

    size_gap = gap.shape[0]
    for i1 in range(size_gap):
        X = gap[i1][0]
        for Y in range(gap[i1][1],gap[i1][2]+1):
            if X%2 == 0:
                xSouth = X+1
                ySouth = N-X+2*Y-1
                if 1 <= N-(ySouth-1) and N-(ySouth-1) <= N and 1 <= xSouth and xSouth <= N:
                    W[end-ySouth,xSouth-1] = 0
                xWest = X+1; yWest = N-X+2*Y
                if 1 <= N-(yWest-1) and N-(yWest-1) <= N and 1 <= xWest and xWest <= N:
                    W[end-yWest,xWest-1] = 0
            elif X%2 == 1:
                xSouth = X; ySouth = N-X+2*Y
                if 1 <= N-(ySouth-1) and N-(ySouth-1) <= N and 1 <= xSouth and xSouth <= N:
                    W[end-ySouth,xSouth-1] = 0
                xWest = X; yWest = N-X+2*Y-1
                if 1 <= N-(yWest-1) and N-(yWest-1) <= N and 1 <= xWest and xWest <= N:
                    W[end-yWest,xWest-1] = 0

    C = algorithm_reduction_weight(W)
    logZnNum = log_partition_function(C)
    logPn = logZnNum - logZnDen
    
    M = shuffling(C)
    fig = draw_dominos(M,edge,dpi,rotated,show_figure)

    return [logPn, logZnNum, logZnDen], fig
