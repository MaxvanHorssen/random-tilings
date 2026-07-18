from numpy import zeros,array,int8,int64,ndarray
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.pyplot import subplots
from numba import njit
from ._draw_aux import _set_color

@njit("(int64, int64, int64, int64, int64)",cache=True,inline="always")
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

@njit("(int64, int64)",cache=True,inline="always")
def type_of_lozenge(x,y):
    # path up
    if x%2 == 1 and y%2 == 0:
        return 0
    # path down
    elif x%2 == 1 and y%2 == 1:
        return 1
    # free of path
    else:
        return 2

@njit("(int64, int64, bool)",cache=True,inline="always")
def points_lozenge(x,y,regular):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 0:
        if regular:
            A = array([[x-1,y-(x-1)//2-2],
                       [x-1,y-(x-1)//2  ],
                       [x+1,y-(x+1)//2+2],
                       [x+1,y-(x+1)//2  ]])
        else:
            A = array([[x-1,y-2],
                       [x-1,y  ],
                       [x+1,y+2],
                       [x+1,y  ]])
    elif type_loz == 1:
        if regular:
            A = array([[x-1,y-(x-1)//2-1],
                       [x-1,y-(x-1)//2+1],
                       [x+1,y-(x+1)//2+1],
                       [x+1,y-(x+1)//2-1]])
        else:
            A = array([[x-1,y-1],
                       [x-1,y+1],
                       [x+1,y+1],
                       [x+1,y-1]])
    else:
        if regular:
            A = array([[x-2,y-x//2  ],
                       [x  ,y-x//2+1],
                       [x+2,y-x//2  ],
                       [x  ,y-x//2-1]])
        else:
            A = array([[x-2,y-1],
                       [x  ,y+1],
                       [x+2,y+1],
                       [x  ,y-1]])
    return A,type_loz

@njit("(int64, int64, bool)",cache=True,inline="always")
def points_path(x,y,regular):
    type_loz = type_of_lozenge(x,y)
    if type_loz == 0:
        if regular:
            A = array([[x-1,y-1-(x-1)//2],[x+1,y+1-(x+1)//2]])
        else:
            A = array([[x-1,y-1],[x+1,y+1]])
    elif type_loz == 1:
        if regular:
            A = array([[x-1,y-(x-1)//2],[x+1,y-(x+1)//2]])
        else:
            A = array([[x-1,y],[x+1,y]])
    else:
        A = array([[0,0],[0,0]])
    return A,type_loz

@njit("(Array(int8, 2, 'C', False, aligned=True), int64, int64, int64, bool, bool)",cache=True)
def compute_hexagon_data(M,A,B,C,regular,paths):
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
                    loz,type_loz = points_path(x,y,regular)
                else:
                    loz,type_loz = points_lozenge(x,y,regular)
                P[k]=loz
                L[k]=type_loz
                k+=1
    return P,L

def draw_lozenges(M,A,B,C,gap=False,edge_width=0,path_width=0,dot_width=0,gap_width=0,
                  shape='regular',color_scheme='standard',dpi=100):
    # Set matplotlib figure layout
    fig,ax = subplots(dpi=dpi)
    margin = 1
    if shape == 'regular':
        ax.set_xlim(-margin,2*(B+C)+margin)
        ax.set_ylim(-B-margin,2*A+C+margin)
        ax.set_aspect(2/(3**0.5))
    elif shape == 'skewed':
        ax.set_xlim(-margin,2*(B+C)+margin)
        ax.set_ylim(-margin,2*(A+C)+margin)
        ax.set_aspect('equal')
    else:
        print('Shape not recognized.')
        raise
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'hexagon',path_width>0)

    # Check for gap
    isgap = type(gap) == ndarray and gap.shape[1]==3

    # Computing data
    P,L = compute_hexagon_data(M,A,B,C,shape=='regular',path_width>0)

    # Plotting
    if path_width > 0:
        if edge_width > 0:
            P0,_ = compute_hexagon_data(M,A,B,C,shape=='regular',False)
            ax.add_collection(PolyCollection(P0,facecolor='None',edgecolor='k',linewidth=edge_width))
        else:
            if shape == 'regular':
                ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*A+C)],[(2*C,2*A+C),(2*(B+C),2*A-B+C)],
                                                [(2*(B+C),2*A-B+C),(2*(B+C),-B+C)],[(2*(B+C),-B+C),(2*B,-B)],[(2*B,-B),(0,0)]],
                                                colors=(0,0,0),linewidths=0.5*path_width))
            else:
                ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*(A+C))],[(2*C,2*(A+C)),(2*(B+C),2*(A+C))],
                                                [(2*(B+C),2*(A+C)),(2*(B+C),2*C)],[(2*(B+C),2*C),(2*B,0)],[(2*B,0),(0,0)]],
                                                colors=(0,0,0),linewidths=0.5*path_width))

        if dot_width > 0:
            I  = L<2
            P0 = P[I]
            L0 = L[I]
            ax.scatter(P0[:,0,0],P0[:,0,1],color=color[L0],linewidth=dot_width,zorder=2)
            ax.scatter(P0[:,1,0],P0[:,1,1],color=color[L0],linewidth=dot_width,zorder=2)

        if isgap and gap_width > 0:
            points_gap_lines = []
            size_gap = gap.shape[0]
            for i1 in range(size_gap):
                x = int(2*gap[i1][0])
                if shape == 'regular':
                    y1 = int(2*gap[i1][1]-0.5*x)
                    y2 = int(2*gap[i1][2]-0.5*x)
                else:
                    y1 = int(2*gap[i1][1])
                    y2 = int(2*gap[i1][2])
                points_gap_lines.append([[x,y1],[x,y2]])
            ax.add_collection(LineCollection(points_gap_lines,colors='r',linewidths=gap_width))
        ax.add_collection(LineCollection(P,colors=color[L],linewidths=path_width,zorder=1))
    else:
        ax.add_collection(PolyCollection(P,facecolor=color[L],edgecolor='k',linewidth=edge_width))
    return fig