from numpy import round, array, int64,int8,zeros,arange
from matplotlib.collections import LineCollection,PolyCollection
from ._draw_aux import _set_color
from numba import njit
from matplotlib.pyplot import subplots
import re

@njit("(int64, int64, int64, int64, int64)",cache=True)
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

@njit("(int64, int64)",cache=True)
def type_of_lozenge(x,y):
    # 'path up'
    if x%2 == 1 and y%2 == 0:
        return 0
    # 'path down'
    elif x%2 == 1 and y%2 == 1:
        return 1
    # 'free of path'
    else:
        return 2

@njit("(int64, int64, bool)",cache=True)
def points_lozenge(x,y,skewed_grid):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 0:
        if skewed_grid:
            A = array([[x-1,y-2],
                       [x-1,y  ],
                       [x+1,y+2],
                       [x+1,y  ]])
        else:
            A = array([[x-1,y-(x-1)//2-2],
                       [x-1,y-(x-1)//2  ],
                       [x+1,y-(x+1)//2+2],
                       [x+1,y-(x+1)//2  ]])
    elif type_loz == 1:
        if skewed_grid:
            A = array([[x-1,y-1],
                       [x-1,y+1],
                       [x+1,y+1],
                       [x+1,y-1]])
        else:
            A = array([[x-1,y-(x-1)//2-1],
                       [x-1,y-(x-1)//2+1],
                       [x+1,y-(x+1)//2+1],
                       [x+1,y-(x+1)//2-1]])
    else:
        if skewed_grid:
            A = array([[x-2,y-1],
                       [x  ,y+1],
                       [x+2,y+1],
                       [x  ,y-1]])
        else:
            A = array([[x-2,y-x//2  ],
                       [x  ,y-x//2+1],
                       [x+2,y-x//2  ],
                       [x  ,y-x//2-1]])
    return A,type_loz

@njit("(int64, int64, bool)",cache=True)
def points_path(x,y,skewed_grid):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 0:
        if skewed_grid:
            A = array([[x-1,y-1],
                       [x+1,y+1]])
        else:
            A = array([[x-1,y-1-(x-1)//2],
                       [x+1,y+1-(x+1)//2]])
    elif type_loz == 1:
        if skewed_grid:
            A = array([[x-1,y],
                       [x+1,y]])
        else:
            A = array([[x-1,y-(x-1)//2],
                       [x+1,y-(x+1)//2]])
    else:
        A = array([[0,0],
                   [0,0]])
    return A,type_loz

@njit("(Array(int8, 2, 'C', False, aligned=True), int64, int64, int64, boolean,boolean)",cache=True)
def compute_hexagon_data(M,A,B,C,skewed_grid,paths):
    N = M.shape[0]
    if paths:
        P = zeros((A*B+A*C+B*C,2,2),dtype=int64)
    else:
        P = zeros((A*B+A*C+B*C,4,2),dtype=int64)
    L = zeros(A*B+A*C+B*C,dtype=int8) 
    k = 0
    for x in range(1,N+1):
        for y in range(1,N+1):
            if in_hexagon(x,y,A,B,C) and M[N-y,x-1] == 1:
                if paths:
                    loz,type_loz = points_path(x,y,skewed_grid)
                else:
                    loz,type_loz = points_lozenge(x,y,skewed_grid)
                P[k]=loz
                L[k]=type_loz
                k+=1

    return P,L






def draw_lozenges(n,M,gap=False,a=1,b=1,c=1,skewed_grid=False,edge=False,
                  paths=False,dots=False,coloring='standard',show_gap=False,
                  dpi=100):
    
    A = int(round(a*n))
    B = int(round(b*n))
    C = int(round(c*n))
    edge = float(edge)

    margin = 1
    fig, ax = subplots(dpi=dpi)
    if skewed_grid:
        ax.set_xlim(-margin,2*(B+C)+margin)
        ax.set_ylim(-margin,2*(A+C)+margin)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(-margin,2*(B+C)+margin)
        ax.set_ylim(-B-margin,2*A+C+margin)
        ax.set_aspect(2/(3**0.5), adjustable='box')
    ax.axis('off')



    if paths:
        N = len(M)
        color = _set_color(coloring,'hexagon',True)
            
        
        if edge >0:
            P,L = compute_hexagon_data(M,A,B,C,skewed_grid,False)
            ax.add_collection(PolyCollection(P,facecolor = 'None',edgecolor='k',linewidth=edge))

        P,L = compute_hexagon_data(M,A,B,C,skewed_grid,True)
        I = L<2
        P0 = P[I]
        L0 = L[I]
        ax.add_collection(LineCollection(P0,colors=color[L0],linewidths=paths,zorder=1))

        if dots:
            ax.scatter(P0[:,0,0],P0[:,0,1],c= color[L0],s=dots,zorder=2)
            ax.scatter(P0[:,1,0],P0[:,1,1],c= color[L0],s=dots,zorder=2)

        if show_gap:
            points_gap_lines = []
            size_gap = gap.shape[0]
            for i1 in range(size_gap):
                x = int(2*gap[i1][0])
                if skewed_grid:
                    y1 = int(2*gap[i1][1])
                    y2 = int(2*gap[i1][2])
                else:
                    y1 = int(2*gap[i1][1]-0.5*x)
                    y2 = int(2*gap[i1][2]-0.5*x)
                points_gap_lines.append([[x,y1],[x,y2]])
            ax.add_collection(LineCollection(points_gap_lines, colors='r', linewidths=paths))

        if edge == 0:
            if skewed_grid:
                ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*(A+C))],[(2*C,2*(A+C)),(2*(B+C),2*(A+C))],
                                                [(2*(B+C),2*(A+C)),(2*(B+C),2*C)],[(2*(B+C),2*C),(2*B,0)],[(2*B,0),(0,0)]],
                                                colors=(0,0,0),linewidths=0.5*paths))
            else:
                ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*A+C)],[(2*C,2*A+C),(2*(B+C),2*A-B+C)],
                                                [(2*(B+C),2*A-B+C),(2*(B+C),-B+C)],[(2*(B+C),-B+C),(2*B,-B)],[(2*B,-B),(0,0)]],
                                                colors=(0,0,0),linewidths=0.5*paths))
    else:

            
        color = _set_color(coloring,'hexagon',False)

        P,C = compute_hexagon_data(M,A,B,C,skewed_grid,False)
        ax.add_collection(PolyCollection(P,facecolor = color[C],edgecolor='k',linewidth=edge))
    return fig