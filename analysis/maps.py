'''
Created on Oct 23, 2020

@author: simon
'''
import os
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import osr
import numpy as np
import colorcet as cc

from plotting import prepare_figure, path_figures
from asymter import path_indices, read_gdal
from kd import read_mask

scenname = 'bandpass'
index = 'logratio'
maxse = 0.02
fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
im, proj, geotrans = read_gdal(fnindex)
fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
selimit = (fnindexse, maxse)
mask = read_mask(selimit=selimit, erosion_iterations=None)
im[np.logical_not(mask)] = np.nan
# extent = (geotrans[0], geotrans[0] + im.shape[1] * geotrans[1],
#           geotrans[3] + im.shape[0] * geotrans[5], geotrans[3])
# ccrsproj = ccrs.Stereographic(central_longitude=-45, true_scale_latitude=70,
#                               central_latitude=90, false_easting=0, false_northing=0)
def gamma(val, exponent=0.5):
    return np.sign(val) * np.abs(val) ** exponent
import matplotlib.path as mpath
import copy
# horrible hack to rotate image while avoiding explicit coordinate conversion
def transform(im):
    return im.T
extent = (-geotrans[3], -geotrans[3] - im.shape[0] * geotrans[5],
          geotrans[0] + im.shape[1] * geotrans[1], geotrans[0])
ccrsproj = ccrs.Stereographic(central_longitude=-135, true_scale_latitude=70,
                              central_latitude=90, false_easting=0, false_northing=0)

pc = ccrs.PlateCarree()
fig, ax = prepare_figure(subplot_kw={'projection': ccrsproj})

ax.set_extent((-180, 180, 59.95, 90), crs=pc)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

cmap = copy.copy(cc.cm['bwy'])
cmap.set_bad('#d0d0d0', 1.)
colw = '#d8dde3'#'#d0d5dd'
vmax = 0.3
ax.imshow(transform(gamma(im)), extent=extent, origin='upper', transform=ccrsproj, 
          vmin=gamma(-vmax), vmax=gamma(vmax), cmap=cmap)
cl = ax.coastlines(color='#000000', zorder=3, lw=0.2, resolution='110m', alpha=0.15)
# ax.add_feature(cartopy.feature.OCEAN, color=colw, zorder=1)
ocean_hr = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
ax.add_feature(ocean_hr, facecolor=colw, edgecolor='none')
ax.add_feature(cfeature.LAKES, facecolor=colw)
ax.add_feature(cfeature.LAKES, facecolor='none', edgecolor='#000000', lw=0.2, alpha=0.15)
gl = ax.gridlines(
    ylocs=(70, 80), xlocs=np.arange(-180, 180, 45), linewidth=0.2, color='#aaaaaa', 
    alpha=0.5)
gl.top_labels = True
gl.left_labels = True
ax.text(-135.2, 80.5, '80', transform=pc, ha='right', va='bottom')
ax.text(-135.2, 70.3, '70', transform=pc, ha='right', va='bottom')
ax.text(0, 58, '0', transform=pc, ha='center', va='center')
ax.text(-45, 57, '-45', transform=pc, ha='center', va='center')
ax.text(-90, 58, '-90', transform=pc, ha='center', va='center')
ax.text(90, 58, '90', transform=pc, ha='center', va='center')
ax.text(135, 57, '135', transform=pc, ha='center', va='center')
ax.text(180, 57, '180', transform=pc, ha='center', va='center')

# plt.show()
fig.savefig(os.path.join(path_figures, 'maptest.pdf'), dpi=600)

