from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from numba import jit
import matplotlib.pyplot as plt
import numpy as np

def draw_lozenges(n,M,a,b,c,edge,dpi,show_figure):
    A = int(np.round(a*n))
    B = int(np.round(b*n))
    C = int(np.round(c*n))
    
    margin = 1
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlim(-margin,2*B+2*C+margin)
    ax.set_ylim(-B-margin,2*A+C+margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_aspect(1/(np.sqrt(3)/2), adjustable='box')

    points_red,points_cyan,points_yellow = comp_poly_points(M,A,B,C)
    ax.add_collection(PatchCollection([Polygon(points) for points in points_red]
                                    ,facecolor=(1,0,0),edgecolor='k',linewidth=edge))
    ax.add_collection(PatchCollection([Polygon(points) for points in points_cyan]
                                    ,facecolor=(0,1,1),edgecolor='k',linewidth=edge))
    ax.add_collection(PatchCollection([Polygon(points) for points in points_yellow]
                                    ,facecolor=(1,1,0),edgecolor='k',linewidth=edge))

    if show_figure:
        plt.show()
    return fig

@jit()
def type_of_lozenge(x,y):
    if x%2 == 1 and y%2 == 0:
        return 1
    elif x%2 == 1 and y%2 == 1:
        return 2
    else:
        return 3

@jit()
def points_lozenge(x,y):
    type_loz2 = type_of_lozenge(x,y)
    if type_loz2 == 1:
        X2 = x+np.array([-1,-1,1,1])
        Y2 = y+np.array([-2,0,2,0])-np.array([(x-1)//2,(x-1)//2,(x+1)//2,(x+1)//2])
    elif type_loz2 == 2:
        X2 = x+np.array([-1,-1,1,1])
        Y2 = y+np.array([-1,1,1,-1])-np.array([(x-1)//2,(x-1)//2,(x+1)//2,(x+1)//2])
    else:
        X2 = x+np.array([-2,0,2,0])
        Y2 = y+np.array([-1,1,1,-1])-np.array([(x-2)//2,x//2,(x+2)//2,x//2])
    return np.round(X2), np.round(Y2)

@jit()
def isinhexagon(x2,y2,A,B,C):
    if x2 <= 2*min(B,C):
        return y2 >= 1 and y2 <= 2*A+x2-1
    elif x2 >= 2*C+1 and x2 <= 2*B:
        return y2 >= 1 and y2 <= 2*(A+C)-1
    elif x2 >= 2*B+1 and x2 <= 2*C:
        return y2 >= x2-2*B+1 and y2 <= 2*A+x2-1
    elif x2 >= 2*max(B,C)+1 and x2 <= 2*(B+C)-1:
        return y2 >= x2-2*B+1 and y2 <= 2*(A+C)-1
    else:
        return False

@jit("(Array(int64, 2, 'C', False, aligned=True), int64, int64, int64)")
def comp_poly_points(M,A,B,C):
    N = M.shape[0]
    P1 = [[(0,0),(0,0)]]
    P2 = [[(0,0),(0,0)]]
    P3 = [[(0,0),(0,0)]]
    for x in range(1,N+1):
        for y in range(1,N+1):
            if isinhexagon(x,y,A,B,C) and M[N-y,x-1] == 1:
                X,Y = points_lozenge(x,y)
                type_loz = type_of_lozenge(x,y)
                if type_loz == 1:
                    P1 += [list(zip(X,Y))]
                elif type_loz == 2:
                    P2 += [list(zip(X,Y))]
                else:
                    P3 += [list(zip(X,Y))]

    return P1[1:],P2[1:],P3[1:]
