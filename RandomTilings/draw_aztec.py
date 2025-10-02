from RandomTilings.weight_aztec import weight_aztec
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_dominos import draw_dominos

def draw_aztec(n,w,edge=0,paths=False,dpi=200,rotated=True,coloring='standard',show_figure=True):
    w = w.astype(float)
    W = weight_aztec(n,w)
    C = algorithm_reduction_weight(W)
    M = shuffling(C)
    fig = draw_dominos(M,edge,paths,dpi,rotated,coloring,show_figure)

    return fig