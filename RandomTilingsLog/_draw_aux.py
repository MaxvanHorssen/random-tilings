from numpy import array,min,max,floating

_colors_A = {'standard' :       array([[1.0,0.0,0.0],
                                       [0.0,1.0,0.0],
                                       [1.0,1.0,0.0],
                                       [0.4,0.4,1.0]]),
             'alternative' :    array([[ 70/255, 60/255, 75/255],
                                       [ 60/255,120/255,155/255],
                                       [190/255,190/255,140/255],
                                       [195/255,130/255, 60/255]]),
              'gray':           array([[0.8,0.8,0.8],
                                       [0.6,0.6,0.6],
                                       [0.2,0.2,0.2],
                                       [0.4,0.4,0.4]]),
              'tropical':       array([[255/255, 33/255,33/255],
                                       [ 31/255, 31/255,255/255],
                                       [ 32/255,255/255, 32/255],
                                       [255/255,255/255, 32/255]]),
              'aztec gray':     array([[6.5/9,6.5/9,6.5/9],
                                       [1.5/9,1.5/9,1.5/9],
                                       [6.0/9,6.0/9,6.0/9],
                                       [2.0/9,2.0/9,2.0/9],
                                       [1.0/9,1.0/9,1.0/9],
                                       [7.0/9,7.0/9,7.0/9],
                                       [2.5/9,2.5/9,2.5/9],
                                       [7.5/9,7.5/9,7.5/9]])}

_colors_H = {'standard' :       array([[1.0,0.0,0.0],
                                       [0.0,1.0,1.0],
                                       [1.0,1.0,0.0]]),
             'alternative' :    array([[  5/255, 50/255,100/255],
                                       [120/255,180/255,200/255],
                                       [135/255, 60/255, 35/255]]),
             'gray' :           array([[0.25,0.25,0.25],
                                       [0.5 ,0.5 ,0.5 ],
                                       [0.75,0.75,0.75]])}


def _rgb_wrapper(coloring):
    C =    array(coloring)
    if C.shape[1]!=3:
        print('Does not satisfy RGB shape input, i.e. (M x 3) array.')
        raise
    dtype = C.dtype.type
    if issubclass(dtype,   floating):
        if    min(C)<0 or    max(C)>1:
            print('Float point RGB colors needs to be between 0 and 1.')
            raise
    else:
        if    min(C)<0 or    max(C)>255:
            print('Integer RGB colors need to be between 0 and 255.')
            raise
        return C/255
    
def _set_color(coloring,model,paths):
    if model == 'aztec':
        colors = _colors_A
    else:
        colors = _colors_H
    if paths:
        if type(coloring)==str and coloring.lower()=='standard':
            return    array([[0.4,0.4,1],[0.4,0.4,1],[0.4,0.4,1]])
        else:
            return _rgb_wrapper(coloring)
    else:
        if type(coloring)==str:
            return colors[coloring]
        else:
            return _rgb_wrapper(coloring)
