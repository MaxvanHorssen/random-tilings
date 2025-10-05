from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import re

def draw_lozenges(n,M,a,b,c,edge,paths,coloring,dpi,show_figure):
    A = int(np.round(a*n))
    B = int(np.round(b*n))
    C = int(np.round(c*n))
    path_scaling = 10/max(A,B,C)

    margin = 1
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlim(-margin,2*(B+C)+margin)
    ax.set_ylim(-B-margin,2*A+C+margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_aspect(2/(3**0.5), adjustable='box') # here

    if paths:
        if coloring == 'standard':
            color_up = (0.4,0.4,1)
            color_down = (0.4,0.4,1)
        else:
            rgb_values = [tuple(float(val)/255 for val in rgb) for rgb in re.findall(r'\((\d+),(\d+),(\d+)\)', coloring)]
            if len(rgb_values) == 2:
                color_up = rgb_values[0]
                color_down = rgb_values[1]
            else:
                print('Invalid coloring')
                return 'Invalid coloring'
        
        points_up,points_down,points_free = compute_points(M,A,B,C,False)
        ax.add_collection(PatchCollection([Polygon(point) for point in points_up]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))
        ax.add_collection(PatchCollection([Polygon(point) for point in points_down]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))
        ax.add_collection(PatchCollection([Polygon(point) for point in points_free]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))

        points_up,points_down,points_free = compute_points(M,A,B,C,True)
        ax.add_collection(LineCollection(points_up,colors=color_up,linewidths=path_scaling))
        ax.add_collection(LineCollection(points_down,colors=color_down,linewidths=path_scaling))

        if edge == 0:
            ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*A+C)],[(2*C,2*A+C),(2*(B+C),2*A-B+C)],
                                              [(2*(B+C),2*A-B+C),(2*(B+C),-B+C)],[(2*(B+C),-B+C),(2*B,-B)],[(2*B,-B),(0,0)]],
                                              colors=(0,0,0),linewidths=0.5*path_scaling))
    else:
        if coloring == 'standard':
            color_up = (1,0,0)
            color_down = (0,1,1)
            color_free = (1,1,0)
        elif coloring == 'alternative':
            color_up = (5/255,50/255,100/255)
            color_down = (120/255,180/255,200/255)
            color_free = (135/255,60/255,35/255)
        elif coloring == 'gray':
            color_up = (0.25,0.25,0.25)
            color_down = (0.5,0.5,0.5)
            color_free = (0.75,0.75,0.75)
        else:
            rgb_values = [tuple(float(val)/255 for val in rgb) for rgb in re.findall(r'\((\d+),(\d+),(\d+)\)', coloring)]
            if len(rgb_values) == 3:
                color_up = rgb_values[0]
                color_down = rgb_values[1]
                color_free = rgb_values[2]
            else:
                print('Invalid coloring')
                return 'Invalid coloring'

        points_up,points_down,points_free = compute_points(M,A,B,C,False)
        ax.add_collection(PatchCollection([Polygon(point) for point in points_up]
                                        ,facecolor=color_up,edgecolor='k',linewidth=edge))
        ax.add_collection(PatchCollection([Polygon(point) for point in points_down]
                                        ,facecolor=color_down,edgecolor='k',linewidth=edge))
        ax.add_collection(PatchCollection([Polygon(point) for point in points_free]
                                        ,facecolor=color_free,edgecolor='k',linewidth=edge))

    if show_figure:
        plt.show()
    return fig

@jit()
def type_of_lozenge(x,y):
    # 'path up'
    if x%2 == 1 and y%2 == 0:
        return 1
    # 'path down'
    elif x%2 == 1 and y%2 == 1:
        return 2
    # 'free of path'
    else:
        return 3

@jit()
def in_hexagon(x,y,A,B,C):
    if x <= 2*min(B,C):
        return y >= 1 and y <= 2*A+x-1
    elif x >= 2*C+1 and x <= 2*B:
        return y >= 1 and y <= 2*(A+C)-1
    elif x >= 2*B+1 and x <= 2*C:
        return y >= x-2*B+1 and y <= 2*A+x-1
    elif x >= 2*max(B,C)+1 and x <= 2*(B+C)-1:
        return y >= x-2*B+1 and y <= 2*(A+C)-1
    else:
        return False

@jit()
def points_path(x,y):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 1:
        X = x+np.array([-1,1])
        Y = y+np.array([-1-(x-1)//2,1-(x+1)//2])
    elif type_loz == 2:
        X = x+np.array([-1,1])
        Y = y+np.array([-(x-1)//2,-(x+1)//2])
    else:
        X = np.array([0,0])
        Y = np.array([0,0])
    return X, Y

@jit()
def points_lozenge(x,y):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 1:
        X = x+np.array([-1,-1,1,1])
        Y = y+np.array([-2-(x-1)//2,-(x-1)//2,2-(x+1)//2,-(x+1)//2])
    elif type_loz == 2:
        X = x+np.array([-1,-1,1,1])
        Y = y+np.array([-1-(x-1)//2,1-(x-1)//2,1-(x+1)//2,-1-(x+1)//2])
    else:
        X = x+np.array([-2,0,2,0])
        Y = y+np.array([-1-(x-2)//2,1-x//2,1-(x+2)//2,-1-x//2])
    return X, Y

@jit("(Array(int64, 2, 'C', False, aligned=True), int64, int64, int64, boolean)")
def compute_points(M,A,B,C,paths):
    N = M.shape[0]
    P1 = [[(0,0),(0,0)]]
    P2 = [[(0,0),(0,0)]]
    P3 = [[(0,0),(0,0)]]
    for x in range(1,N+1):
        for y in range(1,N+1):
            if in_hexagon(x,y,A,B,C) and M[N-y,x-1] == 1:
                if paths:
                    X,Y = points_path(x,y)
                else:
                    X,Y = points_lozenge(x,y)
                type_loz = type_of_lozenge(x,y)
                if type_loz == 1:
                    P1 += [list(zip(X,Y))]
                elif type_loz == 2:
                    P2 += [list(zip(X,Y))]
                else:
                    P3 += [list(zip(X,Y))]
    return P1[1:],P2[1:],P3[1:]