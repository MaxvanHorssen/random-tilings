import numpy as np
import RandomTilings as RT
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import perf_counter as time
from time import sleep
RT.config.log_variant = 1


from numba import njit, prange
from functools import lru_cache


def aztec_image(M, fac=1, *, parallel=False):
    # Return an int8 image, row 0 at the top; no plotting objects created.
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2:
        raise ValueError('M must be a square matrix with even side length')
    if M.dtype.kind not in 'biu':
        raise TypeError('M must contain integers or booleans')
    if isinstance(fac, (bool, np.bool_)) or not isinstance(fac, (int, np.integer)) or fac < 1:
        raise ValueError('fac must be a positive integer')
    # Reuse the usual int8 edge matrix without allocating an active-edge copy.
    if M.dtype != np.int8 and M.dtype != np.bool_:
        M = M == 1
    s = 2*int(fac)
    M = np.ascontiguousarray(M)
    offsets = _offsets(s, s*(len(M)+2))
    if parallel:
        # Finish zeroing before any thread stamps. This avoids concurrent
        # initialization/stamping, and all threads share this one array.
        side = s*(len(M)+2)
        image = np.zeros((side,side),dtype=np.int8)
        _fill_parallel(M,s,offsets,image.reshape(-1))
        return image
    return _stamp(M, s, offsets)


@njit(cache=True)
def _masks(s):
    # Two reusable masks, evaluated at pixel centers. s = 2*fac.
    masks = np.zeros((2, 3*s, 3*s), dtype=np.bool_)
    for r in range(3*s):
        for c in range(3*s):
            # Twice local coordinates in output pixels; v increases UP.
            u = 2*c + 1 - 3*s
            v = 3*s - 2*r - 1
            a, b = u+v, u-v
            # Moving upward increases a and decreases b.
            # Thus a's upper bound and b's lower bound are EXCLUDED.
            # A top edge stays background; a bottom edge gets this tile.
            masks[0,r,c] = (-2*s <= a < 2*s) and (-4*s < b <= 4*s)
            masks[1,r,c] = (-4*s <= a < 4*s) and (-2*s < b <= 2*s)
    return masks


@lru_cache(maxsize=8)
def _offsets(s, side):
    # Option 2: compute only the nonzero pixel positions, once per size/scale.
    # Flatten (row, column) into row*side + column to save address arithmetic.
    masks = _masks(s)
    offsets = np.empty((2, 4*s*s), dtype=np.intp)
    for shape in range(2):
        rows, cols = np.nonzero(masks[shape])
        offsets[shape] = rows*side + cols
    offsets.flags.writeable = False
    return offsets


@njit(cache=True)
def _stamp(M, s, offsets):
    L = M.shape[0]
    side = s*(L+2)
    image = np.zeros((side, side), dtype=np.int8)
    flat = image.reshape(side*side)  # View, not a copy.

    for y in range(L):
        for x in range(L):
            if M[y,x] != 1:
                continue
            shape = 0 if (x+y) % 2 else 1
            if x % 2:
                color = 1 if y % 2 == 0 else 2
            else:
                color = 3 if y % 2 else 4
            # Center: row=s*(y+1.5)-.5, col=s*(x+1.5)-.5.
            # The mask's top-left index is (s*y, s*x).
            base = s*y*side + s*x
            for k in range(offsets.shape[1]):
                flat[base + offsets[shape,k]] = color
    return image


@njit(inline='always')
def _put(flat, base, offsets, color):
    for k in range(offsets.size):
        flat[base+offsets[k]] = color


@njit(cache=True, parallel=True)
def _fill_parallel(M, s, offsets, flat):
    # Each task handles one row of 2x2 blocks. Numba batches those tasks
    # among its threads. Only this OUTER loop uses prange.
    # Requires a valid tiling: nonzero stamp pixels must be disjoint.
    L = M.shape[0]
    side = s*(L+2)
    down = s*side
    for block_row in prange(L//2):
        y = 2*np.int64(block_row)
        for x in range(0,L,2):
            base = s*y*side+s*x
            # Top-left, top-right, bottom-left, bottom-right:
            # shape/color are constants; only edge activity is tested.
            if M[y,x] == 1:
                _put(flat,base,offsets[1],4)
            if M[y,x+1] == 1:
                _put(flat,base+s,offsets[0],1)
            if M[y+1,x] == 1:
                _put(flat,base+down,offsets[0],3)
            if M[y+1,x+1] == 1:
                _put(flat,base+down+s,offsets[1],2)


def plot_aztec(M, fac=1, cmap = None, filename=None, parallel=False):
    # Return (4-by-4-inch figure, full-resolution int8 image).
    # The figure uses a fixed 100 DPI and has no axes margins. Its 400-by-400
    # display resamples the array using nearest-neighbor interpolation.
    # If filename is given, save a PNG at the original array resolution,
    # with DPI metadata for a 4-inch print size. View that PNG at 100% to
    # inspect every original pixel. Screen inches depend on the display,
    # operating-system scaling and notebook/browser zoom.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    image = aztec_image(M, fac, parallel=parallel)
    if cmap == None:
        cmap = ['white', 'red', 'green', 'yellow', '#6666ff']
    cmap = ListedColormap(cmap)
    fig = plt.figure(figsize=(4, 4), dpi=100, layout='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, origin='upper', interpolation='nearest',
              cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_axis_off()
    if filename is not None:
        plt.imsave(filename, image, cmap=cmap, vmin=-0.5, vmax=4.5,
                   origin='upper', format='png', dpi=image.shape[1]/4)
    return fig, image



from vispy import app, scene
from vispy.color import Colormap

def vis_plot(M,cmap = None):
    image = aztec_image(M,1, parallel=0)

    if cmap is None:
        cmap = ['white', 'red', 'green', 'yellow', '#6666ff']

    cmap = Colormap(cmap, interpolation='zero')

    app.use_app('pyside6')

    canvas = scene.SceneCanvas(
        size=(400, 400),
        show=True,
        keys='interactive'
    )

    view = canvas.central_widget.add_view()

    img = scene.visuals.Image(
        image,
        cmap=cmap,
        clim=(-0.5, 4.5),
        interpolation='nearest',
        parent=view.scene
    )

    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.flip = (0, 1, 0)  # Origin upper
    view.camera.set_range(margin=0)

    app.run()


n = 1000
a = 0
b = 1
c = 0
w = np.array([[1,0,0,1],
              [0,1,0,0],
              [0,0,1,0],
              [1,0,0,1]])


A = RT.Aztec(n,w)
A.shuffle()
M = A.get_edge_matrix()

vis_plot(M)