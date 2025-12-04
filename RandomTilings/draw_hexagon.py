from RandomTilings.weight_hexagon import weight_hexagon
from RandomTilings.algorithm_reduction_weight import algorithm_reduction_weight
from RandomTilings.shuffling import shuffling
from RandomTilings.draw_lozenges import draw_lozenges

def draw_hexagon(n,w,a=1,b=1,c=1,skewed_grid=False,edge=0,paths=False,coloring='standard',dpi=200,show_figure=True):
    w = w.astype(float)
    edge = str(edge)
    W = weight_hexagon(n,w,a,b,c)
    C = algorithm_reduction_weight(W)
    M = shuffling(C)
    fig = draw_lozenges(n,M,False,a,b,c,skewed_grid,edge,paths,coloring,False,dpi,show_figure)
    return fig