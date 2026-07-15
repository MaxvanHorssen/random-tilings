from ._weight_aztec import weight_aztec
from ._weight_hexagon import weight_hexagon
from ._core import reduce_weight,reduce_weight_hd,clear_CTower,shuffling,shuffling_hd
from ._draw_dominos import draw_dominos
from ._draw_lozenges import draw_lozenges
from numpy import round,array2string,ndarray,round,ascontiguousarray,int32,array



    
class Config:
    '''Configuration object for specifying and overviewing important settings.
    
    Overview:
    ---------
     - hd_mode      : Enables or disable the right to use the hard drive mode, this instance can be
                      changed using "config.set_hd_mode".
     - warnings     : Enables or disables printed warnings.
     - mpl_backend  : Sets the backend for displaying the matplotlib plots. Options: 
                      "Inline" (Default) and "Interactive". '''
    hd_mode      = False
    warnings     = True
    mpl_backend  = 'Inline'

    def __setattr__(self, name, value):
        if name == 'used_storage':
            object.__setattr__(self, name, value)
            object.__setattr__(self,'free_storage',self.max_storage-self.used_storage)
        elif name == 'hd_mode':
            if self.hd_mode==False and value:
                msg  = 'Warning: The hard drive mode can now be used. This mode is meant to handle large\n'
                msg += 'tilings. However, large tilings require extreme amount of storage, hence all relveant\n'
                msg += 'data will immediately stored on the hard drive. Therefore, it is important to free\n'
                msg += 'the storage after usage. To free the storage use the comand "close", for example if\n'
                msg += '"A" is your random tiling call "A.close()" to free the storage. Also, never use more\n'
                msg += 'then one random tiling with enabled hard drive mode, since they will overwrite each\n'
                msg += 'others data. Consequenctly, they can not be properly shuffeld of plotted.'
                print(msg)
            object.__setattr__(self,name,bool(value))
        elif name == 'warnings':
            if self.warnings == False and value:
                msg  = 'Warnings regarding numerical instabilities are now disabled!'
            object.__setattr__(self,name,bool(value))

        elif name == 'mpl_backend':
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                raise RuntimeError("This function must be run inside IPython/Jupyter.")
            
        
            if value == "Interactive" or value == "interactive":
                value = "Interactive"
                if self.mpl_backend!= value:
                    ip.run_line_magic("matplotlib", "widget")
            elif value == "Inline" or value == "inline":
                value = "Inline"
                if self.mpl_backend!= value:
                    ip.run_line_magic("matplotlib", "inline")
            
            else:
                raise ValueError("mode must be 'interactive', 'inline', or 'notebook'")
            
            object.__setattr__(self,name,value)

        else:
            object.__setattr__(self,name,value)

    def __repr__(self):
        string  = 'Warnings        : ' + ['Disabled','Enabled'][self.warnings]+'\n'
        string += 'Hard Drive Mode : ' + ['Disabled','Enabled'][self.hd_mode]+ '\n'
        string += 'MPL Backend     : ' + self.mpl_backend
        return string
    
    def set_hd_mode(self,val):
        self.hd_mode = bool(val)

    def set_mpl_backend(self,mode):
        """
        mode:
            "interactive" -> Jupyter widget backend
            "inline"      -> Classic inline plots
        """
        self.mpl_backend = mode

config = Config()

class RandomTiling:
    '''Random Tiling object:

    Attached Routines:
     - shuffle : Computes a new instance of the random tiling according to the underlying model.
     - plot    : Creates the plot of the current tiling.
     - close   : Closes the tiling when done with.
     - get_M   : Returns a copy of the underlying edge matrix (advanced).
     - get_C   : Returns a copy of the underlying C-Tower (advanced).
     
    General Procedure:
     shuffle -> plot -> shuffle -> ... -> plot -> close'''
    


    def __init__(self,desc,C,E,plot,_type):
        self.__desc = desc
        self.__C = C
        self.__E = E
        self.__plot = plot
        self.__M = None
        self.__type = _type
        self.fig  = None
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
        '''Creates a new instances of the random tiling.
        '''
        if self.__closed == False:
            if self.__type <0:
                self.__M = shuffling_hd(-self.__type)
            else:
                self.__M,of = shuffling(self.__C,self.__E)
                if of and config.warnings:
                    print('Numerical instability detected: Overflow/Underflow!')
        else:
            print('This random riling is already closed, create a new one!')

    def plot(self,**args):
        '''This function creates a figure containing the plot of the random tiling.
        
        Inputs:
        ---------
         - edge        : float (default = 0); draws edges around the tiles with width 'edge'.
                         Alternative it can be set to 'ede scaling', to set the edge width in
                         comparison to the plot size.
         - paths       : float (default = 0); if bigger than 0, then the random paths associated
                         to the random tiling will be displayed. The linewidth  corresponds to 
                         the value of "paths".
         - dots        : float (default = 0); if bigger tah 0 and "paths">0, then the particles
                         of associated determinental point process will be displayed. The size
                         of the dots is regulated by the value of "dots".
         - coloring    : str or rgb colors, options are 'standard', 'alternative', 'tropical'
                         and 'gray'. For the aztec diamond, we also have the option 'aztec gray'.
         - dpi         : integer (default = 100); resolution of the created plot. 
         - show_gap    : float (default = 0); if bigger than 0, then the gap will be visualized,
                         if a gap was used. The thickness of the visualizting line is given by
                         the numerical value of "show_gap".

         Special:
         ---------
          - For Aztec the option 'orientation' (default = 'diamond'), alternative value 'square'.
          - For Hexagon the option 'skewed_grid' (default = False), which uses a skewed grid.
        '''
        if self.__closed == False:
            if type(self.__M) == type(None):
                print('The Random tiling needs to be shuffeld first!')
            else:
                self.fig = self.__plot(self.__M,**args)
        else:
            print('Random tiling is already closed, create a new one!')

    def close(self):
        '''This function closes the current random tiling and deletes all
        connected data, hence freeing the memory.'''
        self.__closed = True
        self.__desc = 'Closed Random Tiling'
        self.__C = None
        self.__E = None
        self.__M = None
        self.__plot = None
        if self.__type<0:
            clear_CTower()
        self.__type = None
    
    def get_M(self):
        if self.__closed == False:
            if self.__type<0:
                print('Not available for hd-mode.')
            elif type(self.__M)== None:
                print('Random tiling needs to be shuffeld first!')
            else:
                return self.__M.copy()
        else:
            print('This random tiling is already closed, create a new one!')
    def get_CE(self):
        if self.__closed== False:
            if self.__type<0:
                print('Not available for hd-mode.')
            else:
                return self.__C.copy(),self.__E.copy()
        else:
            print('This random tiling is already closed, create a new one!')
            

#################################
# Aztec
#################################

def Aztec(n,w,gap=False,hard_drive_mode=False):
    '''Creates a random aztec diamond.
    Input:
    ---------
     - n               : integer; defines the size of the aztec diamond.
     - w               : 2d numpy.ndarray; the periodic base weight.
     - gap             : 2d numpy.ndarray (default = False); for more information see documentation
     - hard_drive_mode : bool (default = False); if true then routine stores actively most data on
                         the hard drive. The option is intendet for large tilings. Note to clear the
                         storage after, using the comand 'close', see the example below.
                         
    Output: RT (random tiling) object.
    
    Example:
    n = 50
    w = np.array([[1]])
    A = RT.Aztec(n,w)
    A.shuffle()
    A.plot()
    A.close()

    Note: The close comand deletes all relevant information of the randomtiling afterwards freeing the
    memory again. This is important when working with large tiles. Only close the tiling once you are 
    finished creating all wanted plots. After closing no new plots can be created.
    '''

    w  = array(w)
    if gap:
        gap = array(gap)

    desc  = 'Aztec Diamond\n'
    desc += 'n   = '+ str(n)+'\n'
    desc += 'w   = '+array2string(w, precision=2, suppress_small=True).replace('\n','\n'+6*' ')+'\n'
    if type(gap)== ndarray:
        desc += 'gap = ' + array2string(gap, precision=2, suppress_small=True).replace('\n','\n'+6*' ')
    else:
        desc+= 'gap = False'

    def draw_Aztec(M,**kwargs):
        return draw_dominos(M,gap = gap,**kwargs)

    w  = w.astype(complex)
    w1 = w.real
    w2 = w.imag 
    w1 = ascontiguousarray(w1)
    w2 = ascontiguousarray(w2)
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

    if hard_drive_mode:
        if config.hd_mode==False:
            print('Hard drive mode also needs to be enabled via "config.set_hd_mode(True)".')
            return
        _type = -W.shape[0]//2
        C = reduce_weight_hd(W)
        E = None
    else:
        _type = 0
        C,E,of = reduce_weight(W,E)
        if of and config.warnings:
            print('Numerical instability detected: Overflow/Underflow!')
    
    return RandomTiling(desc,C,E,draw_Aztec,_type)



#################################
# Hexagon
#################################


def Hexagon(n,w,a=1,b=1,c=1,gap=False,hard_drive_mode=False):
    '''Creates a random hexagon tiling.
    Input:
    ---------
     - n               : integer; defines the base size of the hexagon tiling.
     - a,b,c           : float (default = 1); sets the dimensions of the tiling.
     - w               : 2d numpy.ndarray; the periodic base weight.
     - gap             : 2d numpy.ndarray (default = False); for more information see documentation
     - hard_drive_mode : bool (default = False); if true then routine stores actively most data on
                         the hard drive. The option is intendet for large tilings. Note to clear the
                         storage after, using the comand 'close', see the example below.
                         
    Output: RT (random tiling) object.
    
    Example:
    n = 50
    w = np.array([[1]])
    A = RT.Hexagon(n,w)
    A.shuffle()
    A.plot()
    A.close()

    Note: The close comand deletes all relevant information of the random tiling afterwards freeing the
    memory again. This is important when working with large tiles. Only close the tiling once you are 
    finished creating all wanted plots. After closing no new plots can be created.
    '''
    w  = array(w)
    if gap:
        gap = array(gap)
    
    desc  = 'Hexagon Tiling\n'
    desc += 'n   = '+str(n)+', a = '+ str(a)+', b = '+str(b)+', c = '+str(c)+ '\n'
    desc += 'w   = '+array2string(w, precision=2, suppress_small=True).replace('\n','\n'+6*' ')+'\n'
    if type(gap)== ndarray:
        desc += 'gap = ' + array2string(gap, precision=2, suppress_small=True).replace('\n','\n'+6*' ')
    else:
        desc+= 'gap = False'

    

    def draw_Hexagon(M,skewed_grid=False,edge=0,paths=False,dots=False,coloring='standard',
                     show_gap=False,dpi=100):
        edge = str(edge)
        fig = draw_lozenges(n,M,gap,a,b,c,skewed_grid,edge,paths,dots,coloring,show_gap,dpi)
        return fig
    w = w.astype(complex)
    w = ascontiguousarray(w)
    W = weight_hexagon(n,w,a,b,c)


    if type(gap)==ndarray:
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


    if hard_drive_mode:
        if config.hd_mode==False:
            print('Hard drive mode also needs to be enabled via "config.set_hd_mode(True)".')
            return
        _type = -W.shape[0]//2
        C = reduce_weight_hd(W)
        E = None
    else:
        _type = 0
        E = ascontiguousarray(W.imag).astype(int32)
        W = ascontiguousarray(W.real)
        C,E,of = reduce_weight(W,E)
        if of and config.warnings:
            print('Numerical instability detected: Overflow/Underflow!')

    return RandomTiling(desc,C,E,draw_Hexagon,_type)

