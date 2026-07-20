from numpy import zeros,array,int8,float64,ndarray,arange
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.pyplot import subplots
from numba import njit
from ._draw_aux import _set_color

@njit("(Array(int8, 2, 'C', False, aligned=True),bool,bool,bool)",cache=True)
def compute_domino_data(M,diamond,aztec_gray,paths):
    N = M.shape[0]
    n = N//2
    if paths:
        P = zeros((n*(n+1),2,2))
    else:
        P = zeros((n*(n+1),4,2))
    C = zeros(n*(n+1),dtype=int8)
    k = 0

    if diamond:
        if paths: 
            A0 = array([[ 0. , 0. ],
                        [ 0. , 0. ]]) # nothing
            A1 = array([[-0.5, 0.5],
                        [ 0.5,-0.5]]) # diagonal down
            A2 = array([[-1. , 0. ],
                        [ 1. ,-0. ]]) # horizontal
            A3 = array([[-0.5,-0.5],
                        [ 0.5, 0.5]]) # diagonal up
        else:
            A0 = array([[ 1., 0.5],
                        [ 1.,-0.5],
                        [-1.,-0.5],
                        [-1., 0.5]])
            A1 = array([[ 0.5, 1.],
                        [-0.5, 1.],
                        [-0.5,-1.],
                        [ 0.5,-1.]])
            A2 = A0
            A3 = A1
    else:
        if paths:
            A0 = array([[ 0., 0.],
                        [ 0., 0.]]) # nothing
            A1 = array([[ 0., 1.],
                        [ 0.,-1.]]) # vertical
            A2 = array([[-1., 1.],
                        [ 1.,-1.]]) # diagonal
            A3 = array([[-1., 0.],
                        [ 1., 0.]]) # horizontal
        else:
            A0 = array([[-1.5, 0.5],
                        [-0.5, 1.5],
                        [ 1.5,-0.5],
                        [ 0.5,-1.5]])
            A1 = array([[-1.5,-0.5],
                        [ 0.5, 1.5],
                        [ 1.5, 0.5],
                        [-0.5,-1.5]])
            A2 = A0
            A3 = A1

    if aztec_gray:
        for x in range(N):
            for y in range(N):
                if diamond:
                    c = array([(x+y)/2,(x-y)/2],dtype=float64)
                else:
                    c = array([x+1,N-y],dtype=float64)
                if M[y,x] == 1:
                    # north
                    if x%2 == 1 and y%2 == 0:
                        P[k] = A0 + c
                        C[k] = int((x+N-y)%4 == 3)
                    # east
                    elif x%2 == 1 and y%2 == 1:
                        P[k] = A1 + c
                        C[k] = 2 + int((x-N+y)%4 == 0)
                    # south
                    elif x%2 == 0 and y%2 == 1:
                        P[k] = A2 + c
                        C[k] = 4 + int((x+N-y)%4 == 3)
                    # west
                    else:
                        P[k] = A3 + c
                        C[k] = 6 + int((x-N+y)%4 == 0)
                    k+= 1
    else:
        for x in range(N):
            for y in range(N):
                if diamond:
                    c = array([(x+y)/2,(x-y)/2],dtype=float64)
                else:
                    c = array([x+1,N-y],dtype=float64)

                if M[y,x] == 1:
                    # north
                    if x%2 == 1 and y%2 == 0:
                        P[k] = A0 + c
                        C[k] = 0
                    # east
                    elif x%2 == 1 and y%2 == 1:
                        P[k] = A1 + c
                        C[k] = 1
                    # south
                    elif x%2 == 0 and y%2 == 1:
                        P[k] = A2 + c
                        C[k] = 2
                    # west
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

def draw_dominos(M,gap=False,edge_width=0,path_width=0,dot_width=0,gap_width=0,
                 orientation='diamond',color_scheme='standard',dpi=100):
    N = M.shape[0]
    n = N//2

    # Set matplotlib figure layout
    fig,ax = subplots(dpi=dpi)
    if orientation == 'diamond':
        ax.set_xlim(-1,len(M))
        ax.set_ylim(-len(M)//2-.5,len(M)//2+.5)
    elif orientation == 'square':
        ax.set_xlim(-1.5,len(M)+2.5)
        ax.set_ylim(-1.5,len(M)+2.5)
    else:
        print('Orientation not recognized.')
        raise
    ax.set_aspect('equal')
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'aztec',path_width>0)

    # Check for gap
    isgap = type(gap) == ndarray and gap.shape[1]==3
    
    # Computing data
    P,C = compute_domino_data(M,orientation=='diamond',color_scheme=='aztec gray',path_width>0)
    
    # Plotting
    if path_width > 0:
        if edge_width > 0:
            P0,_ = compute_domino_data(M,orientation=='diamond',False,False)
            ax.add_collection(PolyCollection(P0,facecolor='None',edgecolor='k',linewidth=edge_width))
        else:
            if orientation=='diamond':
                ax.add_collection(LineCollection([[[k-0.5,k],[k-0.5,k+1],[k+0.5,k+1]] for k in range(n)]+
                                         [[[k-0.5,-k],[k-0.5,-k-1],[k+0.5,-k-1]] for k in range(n)]+
                                         [[[N-k-0.5,k],[N-k-0.5,k+1],[N-k-1.5,k+1]] for k in range(n)]+
                                         [[[N-k-0.5,-k],[N-k-0.5,-k-1],[N-k-1.5,-k-1]] for k in range(n)],
                                                colors='k',linewidths=0.5*path_width))
            else:
                ax.add_collection(LineCollection([[(x+0.5,0.5),(x+1.5,-0.5)] for x in range(0,N,2)]
                                                +[[(x+1.5,-0.5),(x+2.5,0.5)] for x in range(0,N,2)]
                                                +[[(-0.5,y+1.5),(0.5,y+2.5)] for y in range(0,N,2)]
                                                +[[(0.5,y+0.5),(-0.5,y+1.5)] for y in range(0,N,2)]
                                                +[[(x+0.5,N+0.5),(x+1.5,N+1.5)] for x in range(0,N,2)]
                                                +[[(x+1.5,N+1.5),(x+2.5,N+0.5)] for x in range(0,N,2)]
                                                +[[(N+1.5,y+1.5),(N+0.5,y+2.5)] for y in range(0,N,2)]
                                                +[[(N+0.5,y+0.5),(N+1.5,y+1.5)] for y in range(0,N,2)],
                                                colors=(0,0,0),linewidths=0.5*path_width))

        if dot_width > 0:
            I  = C>0
            P0 = P[I]
            C0 = C[I]
            ax.scatter(P0[:,0,0],P0[:,0,1],color=color[C0],linewidth=dot_width)
            if orientation=='diamond':
                ax.scatter(arange(n+1,2*n+1)-0.5,arange(-n+1,1)-0.5,color=color[1],linewidth=dot_width)
            else:
                ax.scatter(arange(2,2*n+2,2),n*[0],color=color[1],linewidth=dot_width)

        if isgap and gap_width > 0:
            points_gap_lines = []
            size_gap = gap.shape[0]
            for i1 in range(size_gap):
                x = gap[i1][0]
                y1 = N-x+2*gap[i1][1]-1
                y2 = N-x+2*gap[i1][2]
                points_gap_lines.append([[x,y1],[x,y2]])
            ax.add_collection(LineCollection(points_gap_lines, colors='r', linewidths=gap_width))
        ax.add_collection(LineCollection(P,colors=color[C],linewidths=path_width))
    else:
        ax.add_collection(PolyCollection(P,facecolor=color[C],edgecolor='k',linewidth=edge_width))
    return fig