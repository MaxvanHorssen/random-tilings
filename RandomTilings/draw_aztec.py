from RandomTilings.weight_aztec import weight_aztec
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_dominos import draw_dominos

def draw_aztec(n,w,edge=0,dpi=200,rotated=True,show_figure=True):
    W = weight_aztec(n,w)
    C = algorithm_reduction_weight(W)
    M = shuffling(C)
    fig = draw_dominos(M,edge,dpi,rotated,show_figure)
    return fig