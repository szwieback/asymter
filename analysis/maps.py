'''
Created on Oct 23, 2020

@author: simon
'''
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import copy
import numpy as np
import colorcet as cc
import geopandas
from string import ascii_lowercase

from analysis.plotting import prepare_figure, path_figures
from asymter import path_indices, read_gdal
from analysis.kd import read_mask
from analysis.paths import fnexplandict

locs = {
    'Kolyma Uplands': (62.3, 150.5),
    'Taymyr': (76.2, 103.0),
    'Melville': (76.0, -116.2),
    'Yukon': (65.0, -147.6),
    'Chantrey': (67.0, -92.0),
    'Ragged': (61.3, -144.6)#(60.4, -144.6)
    }

pc = ccrs.PlateCarree()
# hack!
ccrsproj = ccrs.Stereographic(
    central_longitude=-135, true_scale_latitude=70, central_latitude=90, false_easting=0,
    false_northing=0)
hack_extent = lambda geotrans, im: (-geotrans[3], -geotrans[3] - im.shape[0] * geotrans[5],
                                     geotrans[0] + im.shape[1] * geotrans[1], geotrans[0])
ccrs3413 = ccrs.Stereographic(
    central_longitude=-45, true_scale_latitude=70, central_latitude=90, false_easting=0,
    false_northing=0)
ocean_hr = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
lakes_hr = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')

def gamma(val, exponent=0.70):  # 0.80#0.68
    return np.sign(val) * np.abs(val) ** exponent

# horrible hack to rotate image while avoiding explicit coordinate conversion
def transform(im):
    return im.T

def _draw_panel(
        im, fig, ax, circle, ccrsproj, extent=None, cmap=None, vmax=0.3, vmin=None,
        label=None, ticks=None, ticklabels=None, clabelypos=None):
    colw = '#d8dde3'  # '#d0d5dd'
    cbpos = {'x': 0.03, 'dx': 0.35, 'dy': 0.02, 'labx': 0.5, 'laby':-4.0, 'y':-0.03}
    ax.set_extent((-180, 180, 59.95, 90), crs=pc)
    if clabelypos is None: clabelypos = -2.8
    ax.set_boundary(circle, transform=ax.transAxes)
    if vmin is None:
        vmin = -vmax
    aim = ax.imshow(transform(im), extent=extent, origin='upper', transform=ccrsproj,
                    vmin=vmin, vmax=vmax, cmap=cmap)
    ax.coastlines(color='#000000', zorder=3, lw=0.2, resolution='110m', alpha=0.15)
    ax.add_feature(ocean_hr, facecolor=colw, edgecolor='none')
    ax.add_feature(lakes_hr, facecolor=colw)
    ax.add_feature(
        lakes_hr, facecolor='none', edgecolor='#000000', lw=0.2, alpha=0.15)
    ax.gridlines(
        ylocs=(70, 80), xlocs=np.arange(-180, 180, 45), linewidth=0.2, color='#aaaaaa',
        alpha=0.5)
    for lat, latpos in [(80, 80.5), (70, 70.3)]:
        ax.text(-135.2, latpos, str(lat), transform=pc, ha='right', va='bottom')
    for lon, londpos in [(0, 0), (-45, -1), (-90, 0), (90, 0), (135, -1), (180, -1)]:
        ax.text(lon, 58 + londpos, str(lon), transform=pc, ha='center', va='center')

    pos = ax.properties()['position']
    rect_cax = (pos.x0, pos.y0 + cbpos['y'], pos.x1 - pos.x0, cbpos['dy'])

    cax = fig.add_axes(rect_cax)
    cbar = fig.colorbar(
        aim, cax=cax, orientation='horizontal', ticks=ticks)
    if label is not None:
        cax.text(0.5, clabelypos, label, transform=cax.transAxes, va='baseline', ha='center')
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)
    return ax

def _circle_extent():
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle

def maps(scenname='bandpass', index='logratio', maxse=0.06, fnout=None):
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    im, proj, geotrans = read_gdal(fnindex)
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    mask = read_mask(selimit=selimit, erosion_iterations=None)
    im[np.logical_not(mask)] = np.nan

    extent = hack_extent(geotrans, im)

    fig, axs = prepare_figure(
        nrows=2, ncols=2, figsize=(1.7, 1.7), subplot_kw={'projection': ccrsproj},
        remove_spines=False, hspace=0.37, wspace=0.20, left=0.020, bottom=0.095,
        right=0.980, top=0.990)

    circle = _circle_extent()

    cmap = copy.copy(cc.cm['bwy'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    ticklabels = ['-0.4', '', '-0.2', '-0.1', '0.0', '0.1', '0.2', '', '0.4']
    _draw_panel(
        gamma(im), fig, axs[0, 0], circle, ccrsproj, cmap=cmap, vmax=gamma(ticks[-1]),
        label='asymmetry $a$ [-]', extent=extent, ticks=gamma(ticks), ticklabels=ticklabels)
    for locn, loccor in locs.items():
        axs[0, 0].text(
            loccor[1], loccor[0], locn[0],
            ha='center', va='center', transform=ccrs.Geodetic())
    imse, _, _ = read_gdal(fnindexse)
    imse[np.logical_not(mask)] = np.nan
    cmap = copy.copy(cc.cm['CET_CBL1'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [0.0, 0.02, 0.04]
    gamma_ = lambda x: x
    _draw_panel(
        gamma_(imse), fig, axs[0, 1], circle, ccrsproj, cmap=cmap, vmin=gamma_(ticks[0]),
        vmax=gamma_(ticks[-1]), label='asymmetry standard error [-]', extent=extent,
        ticks=gamma_(ticks), ticklabels=ticks)

    imr, _, _ = read_gdal(fnexplandict['ruggedness'])
    cmap = copy.copy(cc.cm['CET_CBL1'])
    rt = lambda x: np.log10(x)
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [30, 100, 300, 1000]
    _draw_panel(
        rt(imr), fig, axs[1, 0], circle, ccrsproj, cmap=cmap, vmin=rt(20),
        vmax=rt(2000), label='relief $r$ [m]', extent=extent,
        ticks=rt(ticks), ticklabels=ticks)

    imT, _, _ = read_gdal(fnexplandict['temp'])
    cmap = copy.copy(cc.cm['CET_CBL1'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [-15, -10, -5, 0, 5]
    _draw_panel(
        imT, fig, axs[1, 1], circle, ccrsproj, cmap=cmap, vmin=-18, vmax=8,
        label='temperature $T$ [$^{\\circ}\\mathrm{C}$]', extent=extent,
        ticks=ticks, ticklabels=ticks)

    ge = geopandas.read_file(fnexplandict['glacier'])
    gemask = ge.area > (35e3) ** 2
    ge_ = ge.loc[gemask]
    for ax in (axs[1, 0],):
        ax.add_geometries(
            ge_.geometry, crs=ccrs3413, edgecolor='#ffffff', facecolor='none', lw=1.0,
            alpha=1.0, zorder=5)
    for jax, ax in enumerate(axs.flatten()):
        ax.text(
            -0.10, 0.98, ascii_lowercase[jax] + ')', transform=ax.transAxes, ha='left',
            va='baseline')
    if fnout is not None:
        fig.savefig(fnout, dpi=450)

def wind_precip_plot(fnout=None):
    imw, proj, geotrans = read_gdal(fnexplandict['wind'])

    fig, axs = prepare_figure(
        nrows=1, ncols=2, figsize=(1.70, 0.83), subplot_kw={'projection': ccrsproj},
        remove_spines=False, bottom=0.15, top=0.99, left=0.03, right=0.97, wspace=0.20)
    circle = _circle_extent()

    extent = hack_extent(geotrans, imw[1, ...])
#     imw[:, np.logical_not(mask)] = np.nan
    cmap = copy.copy(cc.cm['bwy'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [-8, -4, 0, 4, 8]
    _draw_panel(
        imw[1, ...], fig, axs[0], circle, ccrsproj, cmap=cmap, vmin=-8, vmax=8,
        label='meridional wind [$\\mathrm{m}\\,\\mathrm{s}^{-1}$]', extent=extent,
        ticks=ticks, ticklabels=ticks, clabelypos=-5.1)

    imp, _, _ = read_gdal(fnexplandict['prec'])
#     imw[:, np.logical_not(mask)] = np.nan
    cmap = copy.copy(cc.cm['CET_CBL1'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [100, 300, 1000]
    _draw_panel(
        np.log10(imp), fig, axs[1], circle, ccrsproj, cmap=cmap, vmin=np.log10(50),
        vmax=np.log10(2000), label='precipitation [$\\mathrm{mm}$]', extent=extent,
        ticks=np.log10(ticks), ticklabels=ticks, clabelypos=-5.1)

    if fnout is not None:
        fig.savefig(fnout, dpi=450)

def maps_processing(index0='logratio', maxse=0.02, plot_baseline=True, fnout=None):
    scenname = 'bandpass'
    def _read_scenname(scenname, index=index0):
        fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
        im, proj, geotrans = read_gdal(fnindex)
        return im, proj, geotrans
    im, proj, geotrans = _read_scenname(scenname)
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index0}_se.tif')
    selimit = (fnindexse, maxse)
    mask = read_mask(selimit=selimit, erosion_iterations=None)
    im[np.logical_not(mask)] = np.nan

    extent = hack_extent(geotrans, im)

    fig, axs = prepare_figure(
        nrows=2, ncols=2, figsize=(1.7, 1.7), subplot_kw={'projection': ccrsproj},
        remove_spines=False, hspace=0.37, wspace=0.20, left=0.020, bottom=0.095,
        right=0.980, top=0.990)

    circle = _circle_extent()

    cmap = copy.copy(cc.cm['bwy'])
    cmap.set_bad('#d0d0d0', 1.)
    ticks = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    ticklabels = ['-0.4', '', '-0.2', '-0.1', '0.0', '0.1', '0.2', '', '0.4']
    vmax = ticks[-1]

    im, _, _ = _read_scenname('lowpass')
    _draw_panel(
        gamma(im), fig, axs[0, 1], circle, ccrsproj, cmap=cmap, vmax=gamma(vmax),
        label='lowpass $a$ [-]', extent=extent, ticks=gamma(ticks),
        ticklabels=ticklabels)

    im, _, _ = _read_scenname('bandpass002')
    _draw_panel(
        gamma(im), fig, axs[1, 0], circle, ccrsproj, cmap=cmap, vmax=gamma(vmax),
        label='low threshold $a$ [-]', extent=extent, ticks=gamma(ticks),
        ticklabels=ticklabels)

    im, _, _ = _read_scenname('bandpass008')
    _draw_panel(
        gamma(im), fig, axs[1, 1], circle, ccrsproj, cmap=cmap, vmax=gamma(vmax),
        label='high threshold $a$ [-]', extent=extent, ticks=gamma(ticks),
        ticklabels=ticklabels)

    if plot_baseline:
        _draw_panel(
            gamma(im), fig, axs[0, 0], circle, ccrsproj, cmap=cmap, vmax=gamma(vmax),
            label='baseline $a$ [-]', extent=extent, ticks=gamma(ticks),
            ticklabels=ticklabels)
    else:
        ticks = [-15, -10, -5, 0, 5, 10, 15]
        ticklabels = ticks
        vmax = ticks[-1]
        im, _, _ = _read_scenname('bandpass', index='median')
        im[np.logical_not(mask)] = np.nan
        _draw_panel(
            im, fig, axs[0, 0], circle, ccrsproj, cmap=cmap, vmax=vmax,
            label='bandpass $a_m$ [-]', extent=extent, ticks=ticks,
            ticklabels=ticklabels)

    if fnout is not None:
        fig.savefig(fnout, dpi=450)

if __name__ == '__main__':
    # add: a, b, c, d
    # run with more slope options
    maps(fnout=os.path.join(path_figures, 'maps.pdf'))
#     wind_precip_plot(fnout=os.path.join(path_figures, 'mapwindprecip.pdf'))
#     maps_processing(
#         fnout=os.path.join(path_figures, 'maps_processing.pdf'), plot_baseline=True)
#     maps_processing(
#         fnout=os.path.join(path_figures, 'maps_processing_EW.pdf'), index0='logratioEW')
    