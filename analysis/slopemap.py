'''
Created on Dec 21, 2020

@author: simon
'''

from asymter import read_adem_tile_buffer, slope_bp, save_geotiff, path0, inpaint_mask

import os
import numpy as np

def export_slopemap(tile, bp=(100, None)):
    dem, proj, geotrans, _ = read_adem_tile_buffer(tile=tile)
    dem, mask_gap = inpaint_mask(dem, geotrans, bp=bp)
    slope = slope_bp(dem, proj, geotrans, bp=bp)
    slope[:, mask_gap] = np.nan
    fnout = os.path.join(path0, 'profiles', f'{tile[0]}_{tile[1]}_slope.tif')
    save_geotiff(slope, fnout, geotransform=geotrans, proj=proj)

if __name__ == '__main__':
    tile = (70, 32)#(36, 26)#(47, 14)
    export_slopemap(tile)


