from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import transforms
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import re

def draw_dominos(M,edge,paths,dpi,rotated,coloring,show_figure):
    N = M.shape[0]
    path_scaling = 20/N

    margin = 1    
    fig, ax = plt.subplots(dpi=dpi)

    if rotated:
        ax.set_xlim(-3*N//4-margin+1,3*N//4+margin)
        ax.set_ylim(-margin,3*N//2+margin)
    else:
        ax.set_xlim(-margin,N+1+margin)
        ax.set_ylim(-margin,N+1+margin)
    ax.set_aspect('equal')
    ax.axis('off')

    if paths:
        if coloring == 'standard':
            color_east = (0.4,0.4,1)
            color_south = (0.4,0.4,1)
            color_west = (0.4,0.4,1)
        else:
            rgb_values = [tuple(float(val)/255 for val in rgb) for rgb in re.findall(r'\((\d+),(\d+),(\d+)\)', coloring)]
            if len(rgb_values) == 3:
                color_east = rgb_values[0]
                color_south = rgb_values[1]
                color_west = rgb_values[2]
            else:
                print('Invalid coloring')
                return 'Invalid coloring'

        points_north,points_east,points_south,points_west = compute_points(M,False)
        patch_north = ax.add_collection(PatchCollection([Polygon(points) for points in points_north]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))
        patch_east = ax.add_collection(PatchCollection([Polygon(points) for points in points_east]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))
        patch_south = ax.add_collection(PatchCollection([Polygon(points) for points in points_south]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))
        patch_west = ax.add_collection(PatchCollection([Polygon(points) for points in points_west]
                                        ,facecolor='none',edgecolor='k',linewidth=edge))

        points_north,points_east,points_south,points_west = compute_points(M,True)
        lines_east = ax.add_collection(LineCollection(points_east,colors=color_east,linewidths=path_scaling))
        lines_south = ax.add_collection(LineCollection(points_south,colors=color_south,linewidths=path_scaling))
        lines_west = ax.add_collection(LineCollection(points_west,colors=color_west,linewidths=path_scaling))
    else:
        if coloring == 'aztec gray':
            points_north1,points_north2,points_east1,points_east2,points_south1,points_south2,points_west1,points_west2 = ext_compute_points(M,False)
            patch_north1 = ax.add_collection(PatchCollection([Polygon(points) for points in points_north1]
                                            ,facecolor=(6.5/9,6.5/9,6.5/9),edgecolor='k',linewidth=edge))
            patch_north2 = ax.add_collection(PatchCollection([Polygon(points) for points in points_north2]
                                            ,facecolor=(1.5/9,1.5/9,1.5/9),edgecolor='k',linewidth=edge))
            patch_east1 = ax.add_collection(PatchCollection([Polygon(points) for points in points_east1]
                                            ,facecolor=(6/9,6/9,6/9),edgecolor='k',linewidth=edge))
            patch_east2 = ax.add_collection(PatchCollection([Polygon(points) for points in points_east2]
                                            ,facecolor=(2/9,2/9,2/9),edgecolor='k',linewidth=edge))
            patch_south1 = ax.add_collection(PatchCollection([Polygon(points) for points in points_south1]
                                            ,facecolor=(1/9,1/9,1/9),edgecolor='k',linewidth=edge))
            patch_south2 = ax.add_collection(PatchCollection([Polygon(points) for points in points_south2]
                                            ,facecolor=(7/9,7/9,7/9),edgecolor='k',linewidth=edge))
            patch_west1 = ax.add_collection(PatchCollection([Polygon(points) for points in points_west1]
                                            ,facecolor=(2.5/9,2.5/9,2.5/9),edgecolor='k',linewidth=edge))
            patch_west2 = ax.add_collection(PatchCollection([Polygon(points) for points in points_west2]
                                            ,facecolor=(7.5/9,7.5/9,7.5/9),edgecolor='k',linewidth=edge))
        else:
            if coloring == 'standard':
                color_north = (1,0,0)
                color_east = (0,1,0)
                color_south = (1,1,0)
                color_west = (0.4,0.4,1)
            elif coloring == 'alternative':
                color_north = (70/255,60/255,75/255)
                color_east = (60/255,120/255,155/255)
                color_south = (190/255,190/255,140/255)
                color_west = (195/255,130/255,60/255)
            elif coloring == 'gray':
                color_north = (0.8,0.8,0.8)
                color_east = (0.6,0.6,0.6)
                color_south = (0.2,0.2,0.2)
                color_west = (0.4,0.4,0.4)
            else:
                rgb_values = [tuple(float(val) / 255 for val in rgb) for rgb in re.findall(r'\((\d+),(\d+),(\d+)\)', coloring)]
                if len(rgb_values) == 4:
                    color_north = rgb_values[0]
                    color_east = rgb_values[1]
                    color_south = rgb_values[2]
                    color_west = rgb_values[3]
                else:
                    print('Invalid coloring')
                    return 'Invalid coloring'

            points_north,points_east,points_south,points_west = compute_points(M,False)
            patch_north = ax.add_collection(PatchCollection([Polygon(points) for points in points_north]
                                            ,facecolor=color_north,edgecolor='k',linewidth=edge))
            patch_east = ax.add_collection(PatchCollection([Polygon(points) for points in points_east]
                                            ,facecolor=color_east,edgecolor='k',linewidth=edge))
            patch_south = ax.add_collection(PatchCollection([Polygon(points) for points in points_south]
                                            ,facecolor=color_south,edgecolor='k',linewidth=edge))
            patch_west = ax.add_collection(PatchCollection([Polygon(points) for points in points_west]
                                            ,facecolor=color_west,edgecolor='k',linewidth=edge))

    if rotated:
        tr = transforms.Affine2D().rotate_deg(45) + ax.transData
        if paths:
            lines_east.set_transform(tr)
            lines_south.set_transform(tr)
            lines_west.set_transform(tr)

        if coloring == 'aztec gray':
            patch_north1.set_transform(tr)
            patch_north2.set_transform(tr)
            patch_east1.set_transform(tr)
            patch_east2.set_transform(tr)
            patch_south1.set_transform(tr)
            patch_south2.set_transform(tr)
            patch_west1.set_transform(tr)
            patch_west2.set_transform(tr)
        else:
            patch_north.set_transform(tr)
            patch_east.set_transform(tr)
            patch_south.set_transform(tr)
            patch_west.set_transform(tr)
    
    if show_figure:
        plt.show()
    return fig

@jit()
def type_of_domino(x,y):
    # 'North'
    if x%2 == 0 and y%2 == 0:
        return 1
    # 'East'
    elif x%2 == 0 and y%2 == 1:
        return 2
    # 'South'
    elif x%2 == 1 and y%2 == 1:
        return 3
    # 'West'
    else:
        return 4

@jit()
def points_path(x,y):
    type_domino = type_of_domino(x,y)
    if type_domino == 1:
        X = np.array([0.,0.])
        Y = np.array([0.,0.])
    elif type_domino == 2:
        X = x + np.array([0.,0.])
        Y = y + np.array([1.,-1.])
    elif type_domino == 3:
        X = x + np.array([-1.,1.])
        Y = y + np.array([1.,-1.])
    else:
        X = x + np.array([-1.,1.])
        Y = y + np.array([0.,0.])
    return X, Y

@jit()
def points_domino(x,y):
    type_domino = type_of_domino(x,y)
    if type_domino == 1 or type_domino == 3:
        X = x + np.array([-1.5,-0.5,1.5,0.5])
        Y = y + np.array([0.5,1.5,-0.5,-1.5])
    else:
        X = x + np.array([-1.5,0.5,1.5,-0.5])
        Y = y + np.array([-0.5,1.5,0.5,-1.5])
    return X, Y

@jit("(Array(int64, 2, 'C', False, aligned=True), boolean)")
def compute_points(M,paths):
    N = M.shape[0]
    end = N
    P1 = [[(0.,0.),(0.,0.)]]
    P2 = [[(0.,0.),(0.,0.)]]
    P3 = [[(0.,0.),(0.,0.)]]
    P4 = [[(0.,0.),(0.,0.)]]
    for x in range(1,N+1):
        for y in range(1,N+1):
            if M[end-y,x-1] == 1:
                if paths:
                    X,Y = points_path(x,y)
                else:
                    X,Y = points_domino(x,y)

                type_dom = type_of_domino(x,y)
                if type_dom == 1:
                    P1 += [list(zip(X,Y))]
                elif type_dom == 2:
                    P2 += [list(zip(X,Y))]
                elif type_dom == 3:
                    P3 += [list(zip(X,Y))]
                else:
                    P4 += [list(zip(X,Y))]
    return P1[1:],P2[1:],P3[1:],P4[1:]

@jit()
def ext_type_of_domino(x,y):
    type_domino = type_of_domino(x,y)
    # 'North 1/2'
    if type_domino == 1:
        return 1 + int((x+y)%4 == 0)
    # 'East 1/2'
    elif type_domino == 2:
        return 3 + int((x-y)%4 == 1)
    # 'South 1/2'
    elif type_domino == 3:
        return 5 + int((x+y)%4 == 0)
    # 'West 1/2'
    else:
        return 7 + int((x-y)%4 == 1)

@jit("(Array(int64, 2, 'C', False, aligned=True), boolean)")
def ext_compute_points(M,paths):
    N = M.shape[0]
    end = N
    P1 = [[(0.,0.),(0.,0.)]]
    P2 = [[(0.,0.),(0.,0.)]]
    P3 = [[(0.,0.),(0.,0.)]]
    P4 = [[(0.,0.),(0.,0.)]]
    P5 = [[(0.,0.),(0.,0.)]]
    P6 = [[(0.,0.),(0.,0.)]]
    P7 = [[(0.,0.),(0.,0.)]]
    P8 = [[(0.,0.),(0.,0.)]]
    for x in range(1,N+1):
        for y in range(1,N+1):
            if M[end-y,x-1] == 1:
                if paths:
                    X,Y = points_path(x,y)
                else:
                    X,Y = points_domino(x,y)

                ext_type_dom = ext_type_of_domino(x,y)
                if ext_type_dom == 1:
                    P1 += [list(zip(X,Y))]
                elif ext_type_dom == 2:
                    P2 += [list(zip(X,Y))]
                elif ext_type_dom == 3:
                    P3 += [list(zip(X,Y))]
                elif ext_type_dom == 4:
                    P4 += [list(zip(X,Y))]
                elif ext_type_dom == 5:
                    P5 += [list(zip(X,Y))]
                elif ext_type_dom == 6:
                    P6 += [list(zip(X,Y))]
                elif ext_type_dom == 7:
                    P7 += [list(zip(X,Y))]
                else:
                    P8 += [list(zip(X,Y))]
    return P1[1:],P2[1:],P3[1:],P4[1:],P5[1:],P6[1:],P7[1:],P8[1:]