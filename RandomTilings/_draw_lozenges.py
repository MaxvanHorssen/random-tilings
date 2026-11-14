from numpy import zeros,array,int8,uint8,int64,ndarray,stack,asarray
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.pyplot import subplots
from PIL import Image
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

@njit("(Array(int8, 2, 'C', False, aligned=True), int64, int64, int64, bool, bool)",cache=True)
def compute_hexagon_data(M,A,B,C,regular,paths):
    N = M.shape[0]
    if paths:
        P = zeros((A*B+A*C+B*C,2,2),dtype=int64)
    else:
        P = zeros((A*B+A*C+B*C,4,2),dtype=int64)
    L = zeros(A*B+A*C+B*C,dtype=int8) 
    k = 0

    if regular:
        if paths:
            # path up
            A0 = array([[-1, 0],
                        [ 1, 1]])
            # path down
            A1 = array([[-1, 1],
                        [ 1, 0]])
            # free of path
            A2 = array([[ 0, 0],
                        [ 0, 0]])
        else:
            # path up
            A0 = array([[ 1, 2],
                        [ 1, 0],
                        [-1,-1],
                        [-1, 1]])
            # path down
            A1 = array([[ 1,-1],
                        [-1, 0],
                        [-1, 2],
                        [ 1, 1]])
            # free of path
            A2 = array([[-2, 0],
                        [ 0, 1],
                        [ 2, 0],
                        [ 0,-1]])
    else:
        if paths:
            # path up
            A0 = array([[-1,-1],
                        [ 1, 1]])
            # path down
            A1 = array([[-1, 0],
                        [ 1, 0]])
            # free of path
            A2 = array([[ 0, 0],
                        [ 0, 0]])
        else:
            # path up
            A0 = array([[ 1, 2],
                        [ 1, 0],
                        [-1,-2],
                        [-1, 0]])
            # path down
            A1 = array([[ 1,-1],
                        [-1,-1],
                        [-1, 1],
                        [ 1, 1]])
            # free of path
            A2 = array([[-2,-1],
                        [ 0, 1],
                        [ 2, 1],
                        [ 0,-1]])

    for x in range(1,N+1):
        for y in range(1,N+1):
            if regular:
                c = array([x,y-1-(x-1)//2])
            else:
                c = array([x,y])
            if in_hexagon(x,y,A,B,C) and M[N-y,x-1] == 1:
                # path up
                if x%2 == 1 and y%2 == 0:
                    P[k] = A0 + c
                    L[k] = 0
                # path down
                elif x%2 == 1 and y%2 == 1:
                    P[k] = A1 + c
                    L[k] = 1
                # free of path
                else:
                    P[k] = A2 + c
                    L[k] = 2
                k+= 1

    if paths:
        I = L>=0
        P = P[I]
        L = L[I]
    return P,L

def draw_lozenges(M,A,B,C,border_size=0,boundary_size=0,shape='regular',color_scheme='standard',dpi=100):
    # Set matplotlib figure layout
    fig,ax = subplots(dpi=dpi)
    if shape == 'regular':
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-B-1,2*A+C+1)
        ax.set_aspect(2/(3**0.5))
    elif shape == 'skewed':
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-1,2*(A+C)+1)
        ax.set_aspect('equal')
    else:
        print('Shape not recognized.')
        raise
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'hexagon',False)

    # Computing data
    P,L = compute_hexagon_data(M,A,B,C,shape=='regular',False)

    # Plotting
    ax.add_collection(PolyCollection(P,facecolor=color[L],edgecolor='k',linewidth=border_size))
    if boundary_size > 0:
        if shape == 'regular':
            ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*A+C)],[(2*C,2*A+C),(2*(B+C),2*A-B+C)],
                                            [(2*(B+C),2*A-B+C),(2*(B+C),-B+C)],[(2*(B+C),-B+C),(2*B,-B)],[(2*B,-B),(0,0)]],
                                            colors=(0,0,0),linewidths=boundary_size))
        else:
            ax.add_collection(LineCollection([[(0,0),(0,2*A)],[(0,2*A),(2*C,2*(A+C))],[(2*C,2*(A+C)),(2*(B+C),2*(A+C))],
                                            [(2*(B+C),2*(A+C)),(2*(B+C),2*C)],[(2*(B+C),2*C),(2*B,0)],[(2*B,0),(0,0)]],
                                            colors=(0,0,0),linewidths=boundary_size))
    return fig, shape

def draw_lozenges_dimers(M,A,B,C,fig,shape,line_size=-1,dot_size=-1,dot_border_size=-1):
    N = max(A,B,C)

    # Default scaling sizes
    if line_size == -1:
        line_size = 30/(N+2*N**0.5)
    if dot_size == -1:
        dot_size = 200/N
    if dot_border_size == -1:
        dot_border_size = 10/N

    # Set matplotlib figure layout
    ax = fig.axes[0]
    if shape == 'regular':
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-B-1,2*A+C+1)
        ax.set_aspect(2/(3**0.5))
    else:
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-1,2*(A+C)+1)
        ax.set_aspect('equal')
    ax.axis('off')

    # Computing data
    P,_ = compute_hexagon_data(M,A,B,C,shape=='regular',False)
    P0 = (1/3*P[:,0]+2/3*P[:,2])
    P1 = (2/3*P[:,0]+1/3*P[:,2])
    Q  = stack([P0,P1],axis=1)

    # Plotting
    ax.add_collection(LineCollection(Q,color='k',linewidth=line_size))
    ax.scatter(P0[:,0],P0[:,1],color='k',s=dot_size,facecolor='w',zorder=2,linewidth=dot_border_size)
    ax.scatter(P1[:,0],P1[:,1],color='k',s=dot_size,zorder=2,linewidth=dot_border_size)
    return fig

def draw_lozenges_paths(M,A,B,C,fig,shape,path_size=-1,dot_size=-1,gap=False,gap_size=0,color_scheme='standard'):
    N = max(A,B,C)

    # Default scaling sizes
    if path_size == -1:
        path_size = 30/(N+2*N**0.5)
    if dot_size == -1:
        dot_size = 200/N

    # Set matplotlib figure layout
    ax = fig.axes[0]
    if shape == 'regular':
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-B-1,2*A+C+1)
        ax.set_aspect(2/(3**0.5))
    else:
        ax.set_xlim(-1,2*(B+C)+1)
        ax.set_ylim(-1,2*(A+C)+1)
        ax.set_aspect('equal')
    ax.axis('off')

    # Set color scheme
    color = _set_color(color_scheme,'hexagon',True)

    # Check for gap
    is_gap = type(gap) == ndarray and gap.shape[1]==3

    # Computing data
    P,L = compute_hexagon_data(M,A,B,C,shape=='regular',True)

    # Plotting
    ax.add_collection(LineCollection(P,colors=color[L],linewidths=path_size,zorder=1))
    if dot_size > 0:
        I  = L<2
        P0 = P[I]
        ax.scatter(P0[:,0,0],P0[:,0,1],color=color[-1],s=dot_size,zorder=2)
        ax.scatter(P0[:,1,0],P0[:,1,1],color=color[-1],s=dot_size,zorder=2)
    if is_gap and gap_size > 0:
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
        ax.add_collection(LineCollection(points_gap_lines,colors='r',linewidths=gap_size))
    return fig

##############
# Pixel Plot #
##############

_PIXEL_LEFT = array([
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 0, 0, 0],
], dtype=bool)

_PIXEL_RIGHT = array([
    [1, 0, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 1],
], dtype=bool)

_PIXEL_HORIZONTAL = array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
], dtype=bool)


@njit(cache=True)
def _compute_pixel_hexagon(M, A, B, C):
    image = zeros((2*(2*A + B + C), 4*(B + C)), dtype=uint8)
    N = M.shape[0]
    top = 2*A + C

    for x in range(1, 2*(B + C)):
        for y in range(1, 2*(A + C)):
            if not in_hexagon(x, y, A, B, C):
                continue
            if M[N - y, x - 1] != 1:
                continue

            center_y = y - 1 - (x - 1)//2

            if x % 2:
                if y % 2 == 0:
                    mask, color = _PIXEL_LEFT, 1
                else:
                    mask, color = _PIXEL_RIGHT, 2
                row = 2*(top - center_y - 2)
                col = 2*(x - 1)
            else:
                mask, color = _PIXEL_HORIZONTAL, 3
                row = 2*(top - center_y - 1)
                col = 2*(x - 2)

            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if mask[r, c]:
                        image[row + r, col + c] = color

    return image


def pixel_hexagon(M, A, B, C, filename, color_scheme='standard'):
    """Save a regular hexagon tiling as a palette image."""
    M = asarray(M)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError('M must be a square matrix')
    if M.dtype.kind not in 'biu':
        raise TypeError('M must contain integers or booleans')

    for size in (A, B, C):
        if isinstance(size, bool) or not isinstance(size, (int, int64)):
            raise TypeError('A, B and C must be integers')
        if size < 1:
            raise ValueError('A, B and C must be positive')

    if M.shape[0] < 2*max(B + C, A + C) - 1:
        raise ValueError('M is too small for the given A, B and C')

    colors = asarray(
        _set_color(color_scheme, 'hexagon', False),
        dtype=float,
    )

    # Background, followed by the three lozenge colours.
    palette = zeros((256, 3), dtype=uint8)
    palette[0] = 255
    palette[1:4] = (
        colors[:3, :3].clip(0, 1)*255
    ).round().astype(uint8)

    image = _compute_pixel_hexagon(M, A, B, C)

    img = Image.fromarray(image).convert('P')
    img.putpalette(palette.ravel().tolist())
    img.save(filename, bits=2, optimize=True)