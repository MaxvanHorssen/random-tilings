from RandomTilings.weight_hexagon import weight_hexagon
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_lozenges import draw_lozenges

def draw_hexagon(n,w,a=1,b=1,c=1,edge=0,dpi=200,show_figure=True):
    w = w.astype(float)
    W = weight_hexagon(n,w,a,b,c)
    C = algorithm_reduction_weight(W)
    M = shuffling(C)
    fig = draw_lozenges(n,M,a,b,c,edge,dpi,show_figure)

    return fig
