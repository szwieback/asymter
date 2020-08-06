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

def tile_number(point):
    point = np.array(point)
    tile0 = ceil((point[1, ...] - corner0[0]) / twidth[0])
    tile1 = floor((-corner0[1] - point[0, ...]) / twidth[1])
    return (np.stack((tile0, tile1), axis=0).astype(np.int64))

def create_grid(spacing=spacingdef):
    pos0 = np.arange(corner0[0], corner1[0] - spacing[0], step=spacing[0]) + spacing[0] / 2
    pos1 = np.arange(corner0[1], corner1[1] - spacing[1], step=spacing[1]) + spacing[1] / 2
    tile0_1 = tile_number(np.stack((pos0, 0 * pos0)))[1, ...]
    tile1_0 = tile_number(np.stack((0* pos1, pos1)))[0, ...]
    print(tile0_1)
    print(tile1_0[np.argmin(np.abs(1280000.0-pos1))])
    
if __name__ == '__main__':
    point1 = (-2400000.0, 1300000.0)  # 53 17
    pointin1 = (-2370000.0, 1280000.0)  # 53 17
    point2 = (-2300000.0, 1400000.0)  # 54 18
    point3 = (-1675000.0, 4077000.0)  # 81 24
#     print(tile_number(pointin1))
    create_grid()
