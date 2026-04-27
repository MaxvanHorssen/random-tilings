from .weight_aztec import weight_aztec
from .weight_hexagon import weight_hexagon
from .reduce_weight import reduce_weight,reduce_weight_hd
from .draw_dominos import draw_dominos
from .draw_lozenges import draw_lozenges
from .shuffling import shuffling,shuffling_hd
from numpy import round,array2string,ndarray,round

def format_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.2f} {units[i]}"

def storage_aztec(n):
    bytes = 32*n*(n+1)*(2*n+1)/6
    return bytes
    
def storage_hexa(n,a=1,b=1,c=1):
    N = (int(round(a*n))+int(round(b*n))+int(round(c*n))-1)
    return storage_aztec(N)
    

class data_size:
    def __init__(self,bytes):
        self.bytes = bytes
    def __str__(self):
        return format_bytes(self.bytes)
    def __repr__(self):
        return format_bytes(self.bytes)
    
    def __add__(self,other):
        return data_size(self.bytes+other.bytes)
    def __sub__(self, other):
        return data_size(self.bytes-other.bytes)
    
    def __lt__(self,other):
        if self.bytes<other.bytes:
            return True
        return False
    def __gt__(self,other):
        return other < self
    
class Config:
    def __init__(self):
        self.max_storage  = int(5)
        self.used_storage = data_size(0)
        self.hd_mode      = False

    def __setattr__(self, name, value):
        if name=='max_storage':
            object.__setattr__(self, name, data_size(value*2**30))
        else:
            object.__setattr__(self,name,value)

    def __repr__(self):
        string  = 'Maximum Storage : ' + self.max_storage.__str__() + '\n'
        string += 'Currently Used  : ' + self.used_storage.__str__() + '\n'
        return string


config = Config()

class RT:
    def __init__(self,desc,C,plot,storage,hd_mode):
        self.__desc = desc
        self.__C = C
        self.__plot = plot
        self.__M = None
        self.__hd = hd_mode
        self.storage = storage
        config.used_storage += storage

    def __str__(self):
        return self.__desc
    def __repr__(self):
        return self.__desc
    def __del__(self):
        config.used_storage -= self.storage
        del(self.__desc)
        del(self.__C)
        del(self.__plot)
        del(self.__M)

    def shuffel(self):
        if self.__hd:
            self.__M = shuffling_hd(self.__hd)
        else:
            self.__M = shuffling(self.__C)

    def plot(self,**args):
        if type(self.__M) == type(None):
            print('The Random Tiling needs to be shuffeld first!')
        else:
            return self.__plot(self.__M,**args)


def Aztec(n,w,gap=False,hard_drive_mode=False):
    storage = data_size(storage_aztec(n))
    if config.used_storage + storage > config.max_storage:
        print('Required storage exceeds allowed maximum storage. Increase allowed storage or delete ' \
        'instances of Random Tilings.')
        return
    
    desc = 'Aztec Diamond of size n = '+str(n)+'\nwith Weight\n'+array2string(w, precision=2, suppress_small=True)

    def draw_Aztec(M,edge=0,paths=False,dots=False,rotated=True,coloring='standard',
                   show_gap=False,dpi=100,show_figure=False):
        edge = str(edge)
        return draw_dominos(M,gap,edge,paths,dots,rotated,coloring,show_gap,dpi,show_figure)

    w = w.astype(float)
    W = weight_aztec(n,w)
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
        hard_drive_mode = W.shape[0]//2
        C = reduce_weight_hd(W)
    else:
        C = reduce_weight(W)
    
    return RT(desc,C,draw_Aztec,storage,hard_drive_mode)









def Hexagon(n,w,a=1,b=1,c=1,gap=False,hard_drive_mode=False):
    storage = data_size(storage_hexa(n,a,b,c))
    if config.used_storage + storage > config.max_storage:
        print('Required storage exceeds allowed maximum storage. Increase allowed storage or delete ' \
        'instances of Random Tilings.')
        return
    w = w.astype(float)
    
    desc  = 'Hexagon Tiling\n'
    desc += 'n = '+str(n)+', a = '+ str(a)+', b = '+str(b)+', c = '+str(c)+ '\n'
    desc += 'Size = '+ str(int(n*a)+int(n*b)+int(n*c))+'\n'
    desc += 'Weight\n'
    desc += array2string(w, precision=2, suppress_small=True)

    

    def draw_Hexagon(M,skewed_grid=False,edge=0,paths=False,dots=False,coloring='standard',
                     show_gap=False,dpi=100,show_figure=False):
        edge = str(edge)
        fig = draw_lozenges(n,M,gap,a,b,c,skewed_grid,edge,paths,dots,coloring,show_gap,dpi,show_figure)
        return fig
    
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
        hard_drive_mode = W.shape[0]
        C = reduce_weight_hd(W)
    else:
        C = reduce_weight(W)

    return RT(desc,C,draw_Hexagon,storage,hard_drive_mode)

