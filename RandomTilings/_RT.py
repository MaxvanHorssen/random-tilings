from ._weight_aztec import weight_aztec
from ._weight_hexagon import weight_hexagon
from ._core import reduce_weight,shuffling
from ._draw_dominos import draw_dominos
from ._draw_lozenges import draw_lozenges
from numpy import round,array2string,ndarray,ascontiguousarray,int32,array
    
class Config:
    '''Configuration object for specifying global settings.

    Overview:
    ---------
     - "warnings" enables or disables printed warnings.
     - "mpl_backend" sets the backend for displaying Matplotlib plots. Options: "inline" (default) and "interactive".'''
    warnings     = True
    mpl_backend  = 'inline'

    def __setattr__(self, name, value):
        if name == 'warnings':
            if self.warnings == False and value:
                msg  = 'Warnings regarding numerical instabilities are now disabled!'
                print(msg)
            object.__setattr__(self,name,bool(value))
        elif name == 'mpl_backend':
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                raise RuntimeError("This routine must be run inside IPython/Jupyter notebook.")

            if value.lower() == "interactive":
                value = "interactive"
                if self.mpl_backend != value:
                    ip.run_line_magic("matplotlib", "widget")
            elif value.lower() == "inline":
                value = "inline"
                if self.mpl_backend != value:
                    ip.run_line_magic("matplotlib", "inline")
            else:
                raise ValueError("Mode must be 'interactive', 'inline', or 'notebook'")
            object.__setattr__(self,name,value)
        else:
            object.__setattr__(self,name,value)

    def __repr__(self):
        string  = 'Warnings        : ' + ['Disabled','Enabled'][self.warnings]+'\n'
        string += 'MPL Backend     : ' + self.mpl_backend
        return string

    def set_mpl_backend(self,mode):
        """Mode:
            "interactive" -> Jupyter widget backend
            "inline"      -> Classic inline plots"""
        self.mpl_backend = mode

config = Config()

class RandomTiling:
    '''Random Tiling object:

    Attached routines:
    ------------------
     - "shuffle" samples the Random Tiling object according to the underlying model.
     - "plot" creates a figure containing the plot of the sampled random tiling.
     - "close" closes the Random Tiling object.
     - "get_edge_matrix" returns a copy of the underlying edge matrix.'''

    def __init__(self,desc,C,E,plot):
        self.__desc = desc
        self.__C = C
        self.__E = E
        self.__plot = plot
        self.__M = None
        self.fig = None
        self.__closed = False

    def __str__(self):
        return self.__desc
    def __repr__(self):
        return self.__desc
    def __del__(self):
        del(self.__desc)
        del(self.__C)
        del(self.__E)
        del(self.__plot)
        del(self.__M)

    def shuffle(self):
        '''Samples the Random Tiling object according to the underlying model.'''
        if self.__closed == False:
            self.__M,of = shuffling(self.__C,self.__E)
            if of and config.warnings:
                print('Numerical instability detected: Overflow/Underflow!')
        else:
            print('This Random Tiling object is already closed.')

    def plot(self,**args):
        '''This routine creates a figure containing the plot of the sampled random tiling.

        Inputs:
        -------
         - "edge_width" is the thickness of the border around the domino tiles. By default, edge_width = 0, resulting in no border being drawn.
         - "path_width" displays the non-intersecting path system when set to True. By default, path_width = False.
         - "dot_width" is the thickness of the dots visualizing the particles associated with the non-intersecting path system. This option only takes effect when path_width = True. By default, dot_width = 0.
         - "gap_width" is the thickness of the lines visualizing the gaps. By default, gap_width = 0.
         - "dpi" is the resolution of the figure. By default, dpi = 100.

         Aztec specific:
         - "orientation" gives the orientation of the figure, for which two options are available: "diamond" and "square". By default, orientation = "diamond".
         - "color_scheme" is the color scheme of the figure. Five built-in options are available: "standard", "alternative", "gray", "aztec gray" (custom-made for the 2x2-periodic Aztec diamond), and "tropical". Custom color schemes are specified as [(r,g,b),(r,g,b),(r,g,b),(r,g,b)], where the four RGB triples correspond to the colors of the north, west, south, and east dominoes, respectively. Each color component r, g, and b may be given either as an integer in {0,...,255} or as a real number in [0,1]. By default, color_scheme = "standard".

         Hexagon specific:
         - "shape" gives the shape of the figure, for which two options are available: "regular" and "skewed". By default, shape = "regular".
         - "color_scheme" is the color scheme of the figure. Three built-in options are available: "standard", "alternative", and "gray". Custom color schemes are specified as [(r,g,b),(r,g,b),(r,g,b)], where the three RGB triples correspond to the colors of the left, right, horizontal lozenges, respectively. Each color component r, g, and b may be given either as an integer in {0,...,255} or as a real number in [0,1]. By default, color_scheme = "standard".'''
        if self.__closed == False:
            if type(self.__M) == type(None):
                print('The Random Tiling object needs to be shuffled first!')
            else:
                self.fig = self.__plot(self.__M,**args)
        else:
            print('The Random Tiling object is already closed; create a new instance!')

    def close(self):
        '''This routine closes the current Random Tiling object and deletes all its attributes; freeing the memory it used.'''
        self.__closed = True
        self.__desc = 'Closed Random Tiling object'
        self.__C = None
        self.__E = None
        self.__M = None
        self.__plot = None

    def get_edge_matrix(self):
        if self.__closed == False:
            if type(self.__M) == type(None):
                print('The Random Tiling object needs to be shuffled first!')
            else:
                return self.__M.copy()
        else:
            print('The Random Tiling object is already closed; create a new instance!')

#########
# Aztec #
#########

def Aztec(n,w=[[1]],gap=False):
    '''This routine creates a model of the Aztec diamond.

    Input:
    ------
     - "w" is the weighting on the edge graph, which can be a matrix of any size. We adopt the convention that the weighting is assigned on the bottom left corner of the Aztec diamond, and is then periodically extended to the full Aztec diamond. By default, the uniform weighting is used, i.e., w = [[1]].
     - "gap" must be of the form [[t1,x1,y1],[t2,x2,y2],...,[tm,xm,ym]], which means that at each time tj, there is a vertical gap from xj to yj. The points must satisfy tj in {0,1,...,2n} and xj,yj must be integers. It is allowed to take xj = yj, which represents a single point xj.         

    Output:
    -------
    Random Tiling (RT) object

    Basic example:
    A = RT.Aztec(100)
    A.shuffle()
    A.plot()
    A.close()'''

    w  = array(w)
    if gap:
        gap = array(gap)

    desc  = 'Aztec Diamond tiling\n'
    desc += 'n   = '+ str(n)+'\n'
    desc += 'w   = '+array2string(w, precision=2, suppress_small=True).replace('\n','\n'+6*' ')+'\n'
    if type(gap)== ndarray:
        desc += 'gap = ' + array2string(gap, precision=2, suppress_small=True).replace('\n','\n'+6*' ')
    else:
        desc+= 'gap = False'

    def plot_Aztec(M,**kwargs):
        return draw_dominos(M,gap=gap,**kwargs)

    w  = w.astype(complex)
    w1 = ascontiguousarray(w.real)
    w2 = ascontiguousarray(w.imag)
    W  = weight_aztec(n,w1)
    E  = weight_aztec(n,w2).astype(int32)
    if type(gap)==ndarray:
        N = 2*n
        end = N
        size_gap = gap.shape[0]
        for i1 in range(size_gap):
            X = gap[i1][0]
            for Y in range(gap[i1][1],gap[i1][2]+1):
                if X%2 == 0:
                    xSouth = X+1
                    ySouth = N-X+2*Y-1
                    if 1 <= N-(ySouth-1) and N-(ySouth-1) <= N and 1 <= xSouth and xSouth <= N:
                        W[end-ySouth,xSouth-1] = 0
                    xWest = X+1; yWest = N-X+2*Y
                    if 1 <= N-(yWest-1) and N-(yWest-1) <= N and 1 <= xWest and xWest <= N:
                        W[end-yWest,xWest-1] = 0
                elif X%2 == 1:
                    xSouth = X; ySouth = N-X+2*Y
                    if 1 <= N-(ySouth-1) and N-(ySouth-1) <= N and 1 <= xSouth and xSouth <= N:
                        W[end-ySouth,xSouth-1] = 0
                    xWest = X; yWest = N-X+2*Y-1
                    if 1 <= N-(yWest-1) and N-(yWest-1) <= N and 1 <= xWest and xWest <= N:
                        W[end-yWest,xWest-1] = 0

    C,E,of = reduce_weight(W,E)
    if of and config.warnings:
        print('Numerical instability detected: Overflow/Underflow!')

    return RandomTiling(desc,C,E,plot_Aztec)

###########
# Hexagon #
###########

def Hexagon(n,w=[[1],[1]],a=1,b=1,c=1,gap=False):
    '''This routine creates a model of the Hexagon.

    Input:
    ------
     - "w" is the weighting on the edge graph, which can be a matrix of any size. For a pxq periodic weighting, w must be a matrix of size 2pxq. We adopt the convention that the weighting is assigned on the bottom left corner of the hexagon, and is then periodically extended to the full hexagon. By default, the uniform weighting is used, i.e., w = [[1],[1]].
     - "a", "b", and "c" are the side length multipliers, i.e., the hexagon will be of size (an)x(bn)x(cn). By default, a = b = c = 1.
     - "gap" must be of the form [[t1,x1,y1],[t2,x2,y2],...,[tm,xm,ym]], which means that at each time tj, there is a vertical gap from xj to yj. The points must satisfy tj in {0,1,...,2n} and xj,yj must be half-integers. It is allowed to take xj = yj, which represents a single point xj.

    Output:
    -------
    Random Tiling (RT) object

    Basic example:
    H = RT.Hexagon(100)
    H.shuffle()
    H.plot()
    H.close()'''

    w  = array(w)
    if gap:
        gap = array(gap)

    desc  = 'Hexagon tiling\n'
    desc += 'n   = '+str(n)+', a = '+ str(a)+', b = '+str(b)+', c = '+str(c)+ '\n'
    desc += 'w   = '+array2string(w, precision=2, suppress_small=True).replace('\n','\n'+6*' ')+'\n'
    if type(gap) == ndarray:
        desc += 'gap = ' + array2string(gap, precision=2, suppress_small=True).replace('\n','\n'+6*' ')
    else:
        desc+= 'gap = False'

    def plot_Hexagon(M,**kwargs):
        A = int(round(a*n))
        B = int(round(b*n))
        C = int(round(c*n))
        return draw_lozenges(M,A,B,C,gap=gap,**kwargs)

    w = w.astype(complex)
    w = ascontiguousarray(w)
    W = weight_hexagon(n,w,a,b,c)

    if type(gap) == ndarray:
        A = int(round(a*n))
        B = int(round(b*n))
        C = int(round(c*n))
        N = 2*(A+B+C-1)
        end = N
        size_gap = gap.shape[0]
        for i1 in range(size_gap):
            X = int(2*gap[i1][0])
            for Y in range(int(2*gap[i1][1]),int(2*(gap[i1][2]))+1):
                xUp = X-1
                yUp = Y-1
                if 1 <= N-(yUp-1) and N-(yUp-1) <= N and 1 <= xUp and xUp <= N:
                    W[end-yUp,xUp-1] = 0
                xHorizontal = X-1
                yHorizontal = Y
                if 1 <= N-(yHorizontal-1) and N-(yHorizontal-1) <= N and 1 <= xHorizontal and xHorizontal <= N:
                    W[end-yHorizontal,xHorizontal-1] = 0

    E = ascontiguousarray(W.imag).astype(int32)
    W = ascontiguousarray(W.real)
    C,E,of = reduce_weight(W,E)
    if of and config.warnings:
        print('Numerical instability detected: Overflow/Underflow!')

    return RandomTiling(desc,C,E,plot_Hexagon)