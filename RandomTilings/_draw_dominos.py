from numpy import zeros,array,int8,uint8,float64,ndarray,stack,arange,asarray
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.pyplot import subplots
from PIL import Image, ImageColor
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
            # nothing
            A0 = array([[ 0. , 0. ],
                        [ 0. , 0. ]])
            # diagonal down
            A1 = array([[-0.5, 0.5],
                        [ 0.5,-0.5]])
            # horizontal
            A2 = array([[-1. , 0. ],
                        [ 1. ,-0. ]])
            # diagonal up
            A3 = array([[-0.5,-0.5],
                        [ 0.5, 0.5]])
        else:
            A0 = array([[ 1., 0.5],
                        [ 1.,-0.5],
                        [-1.,-0.5],
                        [-1., 0.5]])
            A1 = array([[ 0.5, 1.],
                        [-0.5, 1.],
                        [-0.5,-1.],
                        [ 0.5,-1.]])
            A2 = array([[-1.,-0.5],
                        [-1., 0.5],
                        [ 1., 0.5],
                        [ 1.,-0.5]])
            A3 = array([[-0.5,-1.],
                        [ 0.5,-1.],
                        [ 0.5, 1.],
                        [-0.5, 1.]])
    else:
        if paths:
            # nothing
            A0 = array([[ 0., 0.],
                        [ 0., 0.]])
            # vertical
            A1 = array([[ 0., 1.],
                        [ 0.,-1.]])
            # diagonal
            A2 = array([[-1., 1.],
                        [ 1.,-1.]])
            # horizontal
            A3 = array([[-1., 0.],
                        [ 1., 0.]])
        else:
            A0 = array([[ 1.5,-0.5],
                        [ 0.5,-1.5],
                        [-1.5, 0.5],
                        [-0.5, 1.5]])
            A1 = array([[ 0.5, 1.5],
                        [ 1.5, 0.5],
                        [-0.5,-1.5],
                        [-1.5,-0.5]])
            A2 = array([[-1.5, 0.5],
                        [-0.5, 1.5],
                        [ 1.5,-0.5],
                        [ 0.5,-1.5]])
            A3 = array([[-0.5,-1.5],
                        [-1.5,-0.5],
                        [ 0.5, 1.5],
                        [ 1.5, 0.5],])

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
        C-= 1
        I = C>=0
        P = P[I]
        C = C[I]
    return P,C

def draw_dominos(M,border_size=0,boundary_size=0,orientation='diamond',color_scheme='standard',dpi=100):
    N = M.shape[0]
    n = N//2

    # Set matplotlib figure layout
    fig,ax = subplots(dpi=dpi)
    if orientation == 'diamond':
        ax.set_xlim(-1,N)
        ax.set_ylim(-n-.5,n+.5)
    elif orientation == 'square':
        ax.set_xlim(-1.5,N+2.5)
        ax.set_ylim(-1.5,N+2.5)
    else:
        print('Orientation not recognized.')
        raise
    ax.set_aspect('equal')
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'aztec',False)

    # Computing data
    P,C = compute_domino_data(M,orientation=='diamond',color_scheme=='aztec gray',False)

    # Plotting
    ax.add_collection(PolyCollection(P,facecolor=color[C],edgecolor='k',linewidth=border_size))
    if boundary_size > 0:
        if orientation=='diamond':
            ax.add_collection(LineCollection([[[k-0.5,k],[k-0.5,k+1],[k+0.5,k+1]] for k in range(n)]+
                                        [[[k-0.5,-k],[k-0.5,-k-1],[k+0.5,-k-1]] for k in range(n)]+
                                        [[[N-k-0.5,k],[N-k-0.5,k+1],[N-k-1.5,k+1]] for k in range(n)]+
                                        [[[N-k-0.5,-k],[N-k-0.5,-k-1],[N-k-1.5,-k-1]] for k in range(n)],
                                            colors='k',linewidths=boundary_size))
        else:
            ax.add_collection(LineCollection([[(x+0.5,0.5),(x+1.5,-0.5)] for x in range(0,N,2)]
                                            +[[(x+1.5,-0.5),(x+2.5,0.5)] for x in range(0,N,2)]
                                            +[[(-0.5,y+1.5),(0.5,y+2.5)] for y in range(0,N,2)]
                                            +[[(0.5,y+0.5),(-0.5,y+1.5)] for y in range(0,N,2)]
                                            +[[(x+0.5,N+0.5),(x+1.5,N+1.5)] for x in range(0,N,2)]
                                            +[[(x+1.5,N+1.5),(x+2.5,N+0.5)] for x in range(0,N,2)]
                                            +[[(N+1.5,y+1.5),(N+0.5,y+2.5)] for y in range(0,N,2)]
                                            +[[(N+0.5,y+0.5),(N+1.5,y+1.5)] for y in range(0,N,2)],
                                            colors=(0,0,0),linewidths=boundary_size))
    return fig, orientation

def draw_dominos_dimers(M,fig,orientation,line_size=-1,dot_size=-1,dot_border_size=-1):
    N = M.shape[0]
    n = N//2

    # Default scaling sizes
    if line_size == -1:
        line_size = 30/(N+2*N**0.5)
    if dot_size == -1:
        dot_size = 200/N
    if dot_border_size == -1:
        dot_border_size = 10/N

    # Set matplotlib figure layout
    ax = fig.axes[0]
    if orientation == 'diamond':
        ax.set_xlim(-1,N)
        ax.set_ylim(-n-.5,n+.5)
    else:
        ax.set_xlim(-1.5,N+2.5)
        ax.set_ylim(-1.5,N+2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Computing data
    P,_ = compute_domino_data(M,orientation=='diamond',False,False)
    P0 = (3/4*(P[:,0]+P[:,1])+1/4*(P[:,2]+P[:,3]))/2
    P1 = (1/4*(P[:,0]+P[:,1])+3/4*(P[:,2]+P[:,3]))/2
    Q  = stack([P0,P1],axis=1)

    # Plotting
    ax.add_collection(LineCollection(Q,color='k',linewidth=line_size))
    ax.scatter(P0[:,0],P0[:,1],color='k',s=dot_size,facecolor='w',zorder=2,linewidth=dot_border_size)
    ax.scatter(P1[:,0],P1[:,1],color='k',s=dot_size,zorder=2,linewidth=dot_border_size)
    return fig

def draw_dominos_paths(M,fig,orientation,path_size=-1,dot_size=-1,gap=False,gap_size=0,color_scheme='standard'):
    N = M.shape[0]
    n = N//2

    # Default scaling sizes
    if path_size == -1:
        path_size = 30/(N+2*N**0.5)
    if dot_size == -1:
        dot_size = 200/N

    # Set matplotlib figure layout
    ax = fig.axes[0]
    if orientation == 'diamond':
        ax.set_xlim(-1,N)
        ax.set_ylim(-n-.5,n+.5)
    else:
        ax.set_xlim(-1.5,N+2.5)
        ax.set_ylim(-1.5,N+2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'aztec',True)

    # Check for gap
    is_gap = type(gap) == ndarray and gap.shape[1]==3

    # Computing data
    P,C = compute_domino_data(M,orientation=='diamond',color_scheme=='aztec gray',True)

    # Plotting
    ax.add_collection(LineCollection(P,colors=color[C],linewidths=path_size,zorder=1))
    if dot_size > 0:
        I  = C>0
        P0 = P[I]
        ax.scatter(P0[:,0,0],P0[:,0,1],color=color[-1],s=dot_size,zorder=2)
        if orientation=='diamond':
            ax.scatter(arange(n+1,N+1)-0.5,arange(-n+1,1)-0.5,color=color[-1],s=dot_size,zorder=2)
        else:
            ax.scatter(arange(2,N+2,2),n*[0],color=color[-1],s=dot_size,zorder=2)
    if is_gap and gap_size > 0:
        points_gap_lines = []
        size_gap = gap.shape[0]
        for i1 in range(size_gap):
            x = gap[i1][0]
            y1 = N-x+2*gap[i1][1]-1
            y2 = N-x+2*gap[i1][2]
            points_gap_lines.append([[x,y1],[x,y2]])
        ax.add_collection(LineCollection(points_gap_lines, colors='r', linewidths=gap_size))
    return fig


##############
# Pixel Plot #
##############

_PIXEL_MASKS = array([
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
    ],
], dtype=bool)


@njit(cache=True)
def _compute_pixel_aztec(M, diamond, aztec_gray):
    N = M.shape[0]
    side = N if diamond else 2*(N + 2)
    image = zeros((side, side), dtype=uint8)

    for y in range(N):
        for x in range(N):
            if M[y, x] != 1:
                continue

            if aztec_gray:
                if x % 2:
                    if y % 2 == 0:  # North
                        color = 1 + int((x + N - y) % 4 == 3)
                    else:           # East
                        color = 3 + int((x - N + y) % 4 == 0)
                else:
                    if y % 2:       # South
                        color = 5 + int((x + N - y) % 4 == 3)
                    else:           # West
                        color = 7 + int((x - N + y) % 4 == 0)
            else:
                if x % 2:
                    color = 1 if y % 2 == 0 else 2
                else:
                    color = 3 if y % 2 else 4

            if diamond:
                row = (N - 1 - x + y)//2
                col = (x + y)//2

                image[row, col] = color
                if (x + y) % 2:
                    image[row, col + 1] = color
                else:
                    image[row + 1, col] = color

            else:
                shape = 0 if (x + y) % 2 else 1

                for r in range(6):
                    for c in range(6):
                        if _PIXEL_MASKS[shape, r, c]:
                            image[2*y + r, 2*x + c] = color

    return image


def pixel_aztec(M, filename, orientation='square',
                color_scheme='standard'):
    """Save an Aztec tiling using a predefined drawing colour scheme."""
    orientation = orientation.lower()
    if orientation not in ('square', 'diamond'):
        raise ValueError("orientation must be 'square' or 'diamond'")

    M = asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2:
        raise ValueError('M must be a square matrix with even side length')
    if M.dtype.kind not in 'biu':
        raise TypeError('M must contain integers or booleans')

    # Same colour selection as draw_dominos.
    colors = asarray(
        _set_color(color_scheme, 'aztec', False),
        dtype=float64,
    )

    # Index 0 is the white background; tile colours start at index 1.
    palette = zeros((256, 3), dtype=uint8)
    palette[0] = 255
    palette[1:len(colors) + 1] = (
        colors[:, :3].clip(0, 1)*255
    ).round().astype(uint8)

    image = _compute_pixel_aztec(
        M,
        orientation == 'diamond',
        color_scheme == 'aztec gray',
    )

    img = Image.fromarray(image).convert('P')
    img.putpalette(palette.ravel().tolist())
    img.save(filename, bits=4, optimize=True)