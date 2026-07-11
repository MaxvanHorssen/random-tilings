import numpy as np

_colors = {'standard' :    np.array([[1.0,0.0,0.0],
                                     [0.0,1.0,0.0],
                                     [1.0,1.0,0.0],
                                     [0.4,0.4,1.0]]),
           'alternative' : np.array([[ 70/255, 60/255, 75/255],
                                     [ 60/255,120/255,155/255],
                                     [190/255,190/255,140/255],
                                     [195/255,130/255, 60/255]]),
            'gray':        np.array([[0.8,0.8,0.8],
                                     [0.6,0.6,0.6],
                                     [0.2,0.2,0.2],
                                     [0.4,0.4,0.4]]),
            'tropical':    np.array([[255/255, 33/255,33/255],
                                     [ 31/255, 31/255,255/255],
                                     [ 32/255,255/255, 32/255],
                                     [255/255,255/255, 32/255]]),
            'aztec gray':  np.array([[6.5/9,6.5/9,6.5/9],
                                     [1.5/9,1.5/9,1.5/9],
                                     [6.0/9,6.0/9,6.0/9],
                                     [2.0/9,2.0/9,2.0/9],
                                     [1.0/9,1.0/9,1.0/9],
                                     [7.0/9,7.0/9,7.0/9],
                                     [2.5/9,2.5/9,2.5/9],
                                     [7.5/9,7.5/9,7.5/9]])}

def _rgb_wrapper(coloring):
    C = np.array(coloring)
    if C.shape[1]!=3:
        print('Does not satisfy RGB shape input, i.e. (M x 3) array.')
        raise
    dtype = C.dtype.type
    if issubclass(dtype,np.floating):
        if np.min(C)<0 or np.max(C)>1:
            print('Float point RGB colors needs to be between 0 and 1.')
            raise
    else:
        if np.min(C)<0 or np.max(C)>255:
            print('Integer RGB colors need to be between 0 and 255.')
            raise
        return C/255
    
def _set_color(coloring,paths):
    if paths:
        if type(coloring)==str and coloring.lower()=='standard':
            return np.array([[0.4,0.4,1],[0.4,0.4,1],[0.4,0.4,1]])
        else:
            return _rgb_wrapper(coloring)
    else:
        if type(coloring)==str:
            return _colors[coloring]
        else:
            return _rgb_wrapper(coloring)
