'''
Created on Aug 6, 2020

@author: simon
'''
from numpy import floor, ceil
import numpy as np

corner0 = (-4000000.0, 4100000.0)
corner1 = (3400000.0, -3400000.0)
stride = (32.0, -32.0)
twidth = (100000.0, -100000.0)

spacingdef = (12.5e3, -12.5e3)

EPSGdef = 3413

def ADEMtile_number(point):
    point = np.array(point)
    tile0 = ceil((point[1, ...] - corner0[0]) / twidth[0])
    tile1 = floor((-corner0[1] - point[0, ...]) / twidth[1])
    return (np.stack((tile0, tile1), axis=0).astype(np.int64))

def create_grid(spacing=spacingdef, corner0=corner0, corner1=corner1):
    pos0 = np.arange(corner0[0], corner1[0] - spacing[0], step=spacing[0]) + spacing[0] / 2
    pos1 = np.arange(corner0[1], corner1[1] - spacing[1], step=spacing[1]) + spacing[1] / 2
    return (pos0, pos1)

def tiles_grid(grid):
    tile0_1 = ADEMtile_number(np.stack((grid[0], 0 * grid[0])))[1, ...]
    tile1_0 = ADEMtile_number(np.stack((0* grid[1], grid[1])))[0, ...]
    return (tile0_1, tile1_0)

def gridtiles(grid):
    from itertools import product
    from collections import namedtuple
    Tilestruc = namedtuple('Tilestruc', ('tile', 'pts', 'ind'))
    tilesg = tiles_grid(grid)
    tilesg_ = (set(tilesg[0]), set(tilesg[1]))

    # loop over tiles
    for tile in product(tilesg_[0], tilesg_[1]):
        ind0 = np.nonzero(tilesg[0] == tile[1])[0]
        ind1 = np.nonzero(tilesg[1] == tile[0])[0]
        pts = product(grid[0][ind0], grid[1][ind1])
        yield Tilestruc(tile=tile, pts=pts, ind=product(ind0, ind1))

if __name__ == '__main__':
    point1 = (-2400000.0, 1300000.0)  # 53 17
    pointin1 = (-2370000.0, 1280000.0)  # 53 17
    point2 = (-2300000.0, 1400000.0)  # 54 18
    point3 = (-1675000.0, 4077000.0)  # 81 24
#     print(tile_number(pointin1))

    grid = create_grid()
    
    
