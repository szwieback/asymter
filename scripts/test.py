'''
Created on Nov 4, 2020

@author: simon
'''
import matplotlib.pyplot as plt
import numpy as np

from asymter.ArcticDEM import read_adem_tile
from asymter.terrain import slope_bp, inpaint_mask, _angle_meridian, _logratio_asymindex
from asymter import Geospatial, match_watermask

bp = (100, 2000)
water_cutoffpct = 25
buffer_water = 2 * bp[0]
dem, proj, geotrans = read_adem_tile(tile=(35, 26))
import time 
start = time.time()
dem, mask_gap = inpaint_mask(dem, geotrans, bp=bp)
print(time.time() - start)
geospatial = Geospatial(shape=dem.shape, proj=proj, geotrans=geotrans)
mask_water = match_watermask(
    geospatial, cutoffpct=water_cutoffpct, buffer=buffer_water)
mask = np.logical_or(mask_water, mask_gap)
ang = _angle_meridian(dem, proj, geotrans)
print(ang * 180 / np.pi)
slope = slope_bp(dem, proj, geotrans, ang=ang, bp=bp)
slope[:, mask] = np.nan
print(_logratio_asymindex(slope))
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
dem[mask] = np.nan
axs[0].imshow(dem)
axs[1].imshow(slope[0, ...], vmin=-0.8, vmax=0.8, cmap='bwr')
plt.show()
