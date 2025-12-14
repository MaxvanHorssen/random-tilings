from RandomTilings.weight_aztec import weight_aztec
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_dominos import draw_dominos

def draw_aztec(n,w,edge=0,paths=False,dots=False,rotated=True,coloring='standard',dpi=200,show_figure=True):
    w = w.astype(float)
    edge = str(edge)
    W = weight_aztec(n,w)
    C = algorithm_reduction_weight(W)
    M = shuffling(C)
    fig = draw_dominos(M,False,edge,paths,dots,rotated,coloring,False,dpi,show_figure)
    return fig