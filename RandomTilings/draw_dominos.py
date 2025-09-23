from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import transforms
from numba import jit
import matplotlib.pyplot as plt
import numpy as np

def draw_dominos(M,edge,dpi,rotated,show_figure):
    N = M.shape[0]
    margin = 1    
    fig, ax = plt.subplots(dpi=dpi)

    if rotated:
        ax.set_xlim(-3*N//4-margin+1,3*N//4+margin)
        ax.set_ylim(-margin,3*N//2+margin)
        ax.set_aspect('equal')
        ax.axis('off')
    else:
        ax.set_xlim(-margin,N+1+margin)
        ax.set_ylim(-margin,N+1+margin)

    points_yellow,points_red,points_blue,points_green = comp_poly_points(M)
    patch_red = ax.add_collection(PatchCollection([Polygon(points) for points in points_yellow]
                                    ,facecolor=(1,1,0),edgecolor='k',linewidth=edge))
    patch_cyan = ax.add_collection(PatchCollection([Polygon(points) for points in points_red]
                                    ,facecolor=(1,0,0),edgecolor='k',linewidth=edge))
    patch_yellow = ax.add_collection(PatchCollection([Polygon(points) for points in points_blue]
                                    ,facecolor=(0,0,1),edgecolor='k',linewidth=edge))
    patch_green = ax.add_collection(PatchCollection([Polygon(points) for points in points_green]
                                    ,facecolor=(0,1,0),edgecolor='k',linewidth=edge))
 
    if rotated:
        tr = transforms.Affine2D().rotate_deg(45) + ax.transData
        patch_red.set_transform(tr)
        patch_cyan.set_transform(tr)
        patch_yellow.set_transform(tr)
        patch_green.set_transform(tr)
    
    if show_figure:
        plt.show()
    return fig

@jit()
def Type_of_domino(x,y):
    if x%2 == 1 and y%2 == 1:
        return 1
    elif x%2 == 0 and y%2 == 0:
        return 2
    elif x%2 == 1 and y%2 == 0:
        return 3
    else:
        return 4

@jit()
def Points_domino(x,y):
    type_domino = Type_of_domino(x,y)
    s2 = 2
    if type_domino == 1 or type_domino == 2:
        X2 = x + np.array([-0.5-0.5*s2,-0.5,0.5+0.5*s2,0.5])
        Y2 = y + np.array([0.5,0.5+0.5*s2,-0.5,-0.5-0.5*s2])
    else:
        X2 = x + np.array([-0.5-0.5*s2,0.5,0.5+0.5*s2,-0.5])
        Y2 = y + np.array([-0.5,0.5+0.5*s2,0.5,-0.5-0.5*s2])
    return X2, Y2

@jit("(Array(int64, 2, 'C', False, aligned=True),)")
def comp_poly_points(M):
    N = M.shape[0]
    end = N
    P1 = [[(0.,0.),(0.,0.)]]
    P2 = [[(0.,0.),(0.,0.)]]
    P3 = [[(0.,0.),(0.,0.)]]
    P4 = [[(0.,0.),(0.,0.)]]
    for x in range(1,N+1):
        for y in range(1,N+1):
            if M[end-y,x-1] == 1:
                X,Y = Points_domino(x,y)
                type_loz = Type_of_domino(x,y)
                if type_loz == 1:
                    P1 += [list(zip(X,Y))]
                elif type_loz == 2:
                    P2 += [list(zip(X,Y))]
                elif type_loz == 3:
                    P3 += [list(zip(X,Y))]
                else:
                    P4 += [list(zip(X,Y))]            
    return P1[1:],P2[1:],P3[1:],P4[1:]