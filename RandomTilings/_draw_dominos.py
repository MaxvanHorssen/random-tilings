from matplotlib.collections import PolyCollection, LineCollection
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
from ._draw_aux import _set_color

@njit("(Array(int8, 2, 'C', False, aligned=True),boolean,boolean,boolean)",cache=True)
def compute_domino_data(M,diamond,ac_gray,paths):
    N = M.shape[0]
    n = N//2
    if paths:
        P = np.zeros((n*(n+1),2,2))
    else:
        P = np.zeros((n*(n+1),4,2))
    C = np.zeros(n*(n+1),dtype=np.int8)
    k = 0

    if diamond:
        if paths : 
            A0 = np.array([[0.,0.],
                           [0.,0.]])
            A1 = np.array([[-0.5,0.5],
                           [0.5,-0.5]]) # diagonal down
            A2 = np.array([[-1.,0.],
                           [1.,-0.]])  # horizontal
            A3 = np.array([[-0.5,-0.5],
                           [0.5,0.5]]) # diagonal up
        else:
            A0 = np.array([[1,0.5],
                           [1,-0.5],
                           [-1,-0.5],
                           [-1,0.5]])

            A1 = np.array([[0.5,1],
                           [-0.5,1],
                           [-0.5,-1],
                           [0.5,-1]])
            
            A2 = A0
            A3 = A1
    else:
        if paths:
            A0 = np.array([[0.,0.],
                           [0.,0.]])  # nothing
            A1 = np.array([[0.,1.],
                           [0.,-1.]]) # vertical
            A2 = np.array([[-1.,1.],
                           [1.,-1.]]) # diagonal
            A3 = np.array([[-1.,0.],
                           [1,0.]])   # horizontal
        else:
            A0 = np.array([[-1.5,0.5],
                           [-0.5,1.5],
                           [1.5,-0.5],
                           [0.5,-1.5]])
            A1 = np.array([[-1.5,-0.5],
                           [0.5,1.5],
                           [1.5,0.5],
                           [-0.5,-1.5]])
            A2 = A0
            A3 = A1

    
    if ac_gray:
        for x in range(N):
            for y in range(N):
                if diamond:
                    c = np.array([(x+y)/2,(x-y)/2],dtype=np.float64)
                else:
                    c = np.array([x+1,N-y],dtype=np.float64)
                if M[y,x] == 1:
                    # 'North'
                    if x%2 == 1 and y%2 == 0:
                        P[k] = A0 + c
                        C[k] = int((x+N-y)%4 == 3)
                    # 'East'
                    elif x%2 == 1 and y%2 == 1:
                        P[k] = A1 + c
                        C[k] = 2 + int((x-N+y)%4 == 0)
                    # 'South'
                    elif x%2 == 0 and y%2 == 1:
                        P[k] = A2 + c
                        C[k] = 4 + int((x+N-y)%4 == 3)
                    # 'West'
                    else:
                        P[k] = A3 + c
                        C[k] = 6 + int((x-N+y)%4 == 0)
                    k+= 1

    else:
        for x in range(N):
            for y in range(N):
                if diamond:
                    c = np.array([(x+y)/2,(x-y)/2],dtype=np.float64)
                else:
                    c = np.array([x+1,N-y],dtype=np.float64)

                if M[y,x] == 1:
                    # 'North'
                    if x%2 == 1 and y%2 == 0:
                        P[k] = A0 + c
                        C[k] = 0
                    # 'East'
                    elif x%2 == 1 and y%2 == 1:
                        P[k] = A1 + c
                        C[k] = 1
                    # 'South'
                    elif x%2 == 0 and y%2 == 1:
                        P[k] = A2 + c
                        C[k] = 2
                    # 'West'
                    else:
                        P[k] = A3 + c
                        C[k] = 3
                    k+= 1

    if paths:
        C -= 1
        I  = C>=0
        P  = P[I]
        C  = C[I]
    return P,C



def draw_dominos(M,gap=False,edge=0,paths=False,dots=False,
                 orientation='diamond',coloring='standard',show_gap=False,dpi=100):
    edge =  float(edge)
    paths = float(paths)
    dots  = float(dots)

    ############################################################
    # Setup
    #-----------------------------------------------------------
    # Setting matplotlib-figure layout.
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')

    # Setting color theme
    color = _set_color(coloring,paths)

    # Normalize orientation
    orientation = orientation.lower()

    # Check for aztec gray
    ac_gray = str(coloring).lower()=='aztec gray'

    # Check for diamond shape
    orientation = orientation.lower()
    diamond = orientation=='diamond'

    # Check for gap
    isgap = type(gap) ==np.ndarray and gap.shape[1]==3
    
    ############################################################
    # Computing data + Plotting
    #-----------------------------------------------------------

    P, C = compute_domino_data(M,diamond,ac_gray,paths>0)
    if orientation=='diamond':
        ax.set_xlim(-1,len(M))
        ax.set_ylim(-len(M)//2-.5,len(M)//2+.5)
    elif orientation =='square':
        ax.set_xlim(-1.5,len(M)+2.5)
        ax.set_ylim(-2.5,len(M)+1.5)
    else:
        print('Orientation not recognized.')
        raise


    if paths:
        if edge:
            P0,_ = compute_domino_data(M,diamond,False,False)
            ax.add_collection(PolyCollection(P0,facecolor='None',edgecolor='k',linewidth=edge))
        if dots:
            if diamond:
                shift = np.array([[-0.5,0.5],[-1,0],[-0.5,-0.5]])
                P0 = P[:,0,:]-shift[C]
            else:
                shift = np.array([[0,1],[-1,1],[-1,0]])
                P0 = P[:,0,:]-shift[C]
            ax.scatter(P0[:,0],P0[:,1],color=color[C],linewidth=dots)
        if isgap and show_gap>0:
            points_gap_lines = []
            size_gap = gap.shape[0]
            for i1 in range(size_gap):
                N = M.shape[0]
                x = gap[i1][0]
                y1 = N-x+2*gap[i1][1]-1
                y2 = N-x+2*gap[i1][2]
                points_gap_lines.append([[x,y1],[x,y2]])
            print(points_gap_lines)
            ax.add_collection(LineCollection(points_gap_lines, colors='r', linewidths=show_gap))


        ax.add_collection(LineCollection(P,colors=color[C],linewidths=paths))
        
    else:
        ax.add_collection(PolyCollection(P,facecolor=color[C],edgecolor='k',linewidth=edge))
    return fig