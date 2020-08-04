'''
Created on Aug 2, 2020

@author: simon
'''

import os
from osgeo import gdal, osr
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import warnings
versiondef = '3.0'
resdef = '32m'
invalid = -9999.0,
# proj corresponds to EPSG 3413
projdef = (
    'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
    'AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],'
    'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
lonoffset = float(projdef[projdef.rindex('central_meridian') + 18:].split(']')[0])
pathADEM = '/home/simon/Work/asymter/ArcticDEM/32m'

def read_gdal(fntif):
    f = gdal.Open(fntif, gdal.GA_ReadOnly)
    im = f.ReadAsArray()
    proj = f.GetProjection()
    geotrans = f.GetGeoTransform()
    return im, proj, geotrans

def read_tile(tile=(52, 19), path=pathADEM, res=resdef, version=versiondef):
    tilestr = f'{tile[0]}_{tile[1]}_{res}_v{version}'
    fntif = os.path.join(path, tilestr, tilestr + '_reg_dem.tif')
    dem, proj, geotrans = read_gdal(fntif)
    return dem, proj, geotrans

def zero_pad(arr, pct=25):
    shapeout = (np.array(arr.shape) * (1 + pct / 100))
    shapeout = 2 ** (np.floor(np.log2(shapeout)) + 1)
    shapeout = tuple(shapeout.astype(np.uint32))
    arr_zp = np.zeros(shapeout, dtype=arr.dtype)
    arr_zp[0:arr.shape[0], 0:arr.shape[1]] = arr
    return arr_zp

def spatial_frequencies(shape, samp):
    sf = []
    for n_samples in shape:
        if np.mod(n_samples, 2) != 0: raise NotImplementedError('odd samples')
        sf_dim = np.arange(-n_samples // 2, n_samples // 2) / (n_samples * samp)
        assert(len(sf_dim) == n_samples)
        sf.append(sf_dim)
    return tuple(sf)

def _angle_meridian(proj, geotrans):
    assert proj == projdef
    assert geotrans[2] == geotrans[4] == 0
    xm = geotrans[0] + geotrans[1] * dem.shape[1] // 2
    ym = geotrans[3] + geotrans[5] * dem.shape[0] // 2
    ang = np.arctan2(-xm, -ym)
    return ang

def _gaussian(sfreq, e_folding):
    # e_folding is given as length/period rather than frequency
    assert len(sfreq) == 2
    p0 = (sfreq[0][:, np.newaxis] * e_folding) ** 2
    p1 = (sfreq[1][np.newaxis, :] * e_folding) ** 2
    return np.exp(-(p0 + p1))

def bandpass(dem, bp=(None, None), zppct=25.0, freqdomain=False):
    # bp are e-folding half scales
    speczpc = fftshift(fft2(zero_pad(dem, pct=zppct)))
    sfreq = spatial_frequencies(speczpc.shape, geotrans[1])
    spechpc, speclpc = np.ones_like(speczpc), np.ones_like(speczpc)
    if bp[1] is not None:
        spechpc -= _gaussian(sfreq, bp[1])
    if bp[0] is not None:
        speclpc *= _gaussian(sfreq, bp[0])
    speczpc *= (spechpc * speclpc)
    if freqdomain:
        return speczpc, sfreq
    else:
        dem_ = np.real(ifft2(ifftshift(speczpc)))
        dem_ = dem_[0:dem.shape[0], 0:dem.shape[1]].copy()
        return dem_

def slope_bp(dem, proj, geotrans, bp=(100, 2500), ang=None, zppct=25.0):
    if ang is None:
        ang = _angle_meridian(proj, geotrans)
    speczpc, sfreq = bandpass(dem, bp, zppct=zppct, freqdomain=True)
    specs = np.stack((
        (np.cos(ang) * sfreq[0][:, np.newaxis] - np.sin(ang) * sfreq[1][np.newaxis, :]),
        (-np.sin(ang) * sfreq[0][:, np.newaxis] + np.cos(ang) * sfreq[1][np.newaxis, :])),
        axis=0).astype(np.complex128)
    specs *= 2j * np.pi
    slopezp = np.real(
        ifft2(ifftshift(speczpc[np.newaxis, ...] * specs, axes=(1, 2)), axes=(1, 2)))
    slope = slopezp[:, 0:dem.shape[0], 0:dem.shape[1]]
    return slope

def slope_discrete_bp(
        dem, proj, geotrans, bp=(None, None), ang=None, zppct=25.0):
    if ang is None:
        ang = _angle_meridian(proj, geotrans)
    if bp != (None, None):
        dem_ = bandpass(dem, bp=bp, zppct=zppct)
    else:
        dem_ - dem
    sloped1 = dem_.copy()
    sloped1[:-1, :] -= sloped1[1:, :]
    sloped1 /= geotrans[5]
    sloped2 = dem_.copy()
    sloped2[:, :-1] -= sloped2[:, 1:]
    sloped2 /= geotrans[1]
    nsloped = np.cos(ang) * sloped1 + np.sin(ang) * sloped2
    esloped = -np.sin(ang) * sloped1 + np.cos(ang) * sloped2
    return np.stack((nsloped, esloped), axis=0)

def _median_asymindex(slope):
    slopep = np.nanpercentile(slope[0, ...], (25, 50, 75))
    asymi = -100 * slopep[1] / (slopep[2] - slopep[0])  # pos: N facing steeper
    return asymi

def _logratio_asymindex(slope, minslope=0.05, aspthresh=1):
    slope_ = np.reshape(slope.copy(), (slope.shape[0], -1))
    slope_abs = np.sqrt(np.sum(slope_ ** 2, axis=0))
    asp = slope_[0] / np.abs(slope_[1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        slope_abs_n = slope_abs[np.logical_and(asp >= aspthresh, slope_abs >= minslope)]
        slope_abs_s = slope_abs[np.logical_and(asp <= -aspthresh, slope_abs >= minslope)]
    asymi = np.log10(np.median(slope_abs_n) / np.median(slope_abs_s))
    return asymi

def asymindex(slope, indtype='median', **kwargs):
    if indtype == 'median':
        asymi = _median_asymindex(slope)
    elif indtype == 'logratio':
        asymi = _logratio_asymindex(slope, **kwargs)
    return asymi

def inpaint_mask(dem, invalid=invalid):
    # to do: use opencv inpainting; reasonable length scales
    dem[dem <= invalid] = np.nan  # improve infilling
    from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure
    demf = median_filter(dem, size=2 * int(np.abs(bp[0] / geotrans[1])))
    dem[np.isnan(dem)] = demf[np.isnan(dem)]
    mask = np.isnan(dem)
    l_int = bp[1] if bp[1] is not None else 3 * bp[0]
    selem = generate_binary_structure(2, 1)
    mask_slope = binary_dilation(
        mask, selem, iterations=int(np.abs(l_int / geotrans[1])))
    dem[mask] = np.nanmedian(dem)
    return dem, mask_slope

if __name__ == '__main__':
    tile = (49, 20)  # (52, 19)  # (49, 20)
    dem, proj, geotrans = read_tile(tile=tile)
    bp = (100, 2000)  # 3000 and larger have issues with large-scale slopes
    ang = _angle_meridian(proj, geotrans)

#     ang = np.deg2rad(90)
#     dem = np.zeros((4096, 4096))
#     dem[:, :] = 100 * np.real(np.exp(1j * (np.cos(ang) * np.arange(dem.shape[0])[:, np.newaxis] - np.sin(ang) * np.arange(dem.shape[1])[np.newaxis, :]) / 20))

    dem, mask_slope = inpaint_mask(dem)
    slope = slope_bp(dem, proj, geotrans, bp=bp, ang=ang)
    slope[:, mask_slope] = np.nan

    sloped = slope_discrete_bp(dem, proj, geotrans, ang=ang, bp=bp)
    sloped[:, mask_slope] = np.nan

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=3)
#     dem = bandpass(dem, bp=bp)
    ax[0].imshow(dem)
    vlim = 0.3
    ax[1].imshow(slope[0, ...], vmin=-vlim, vmax=vlim, cmap='bwr')
    ax[2].imshow(sloped[0, ...], vmin=-vlim, vmax=vlim, cmap='bwr')

#     nslopew = nslope[1700:2400, 1250:2500]
    for slope_ in (slope, sloped):
        slopew = slope_[:, 1000:-1000, 1000:-1000]
        print(asymindex(slopew), asymindex(slopew, indtype='logratio'))
        print(np.nanmean(slopew[0, ...]))  # also store the mean/median

    plt.show()

    # to do: test EW, aspect, aspect mask

#     src = osr.SpatialReference()
#     tgt = osr.SpatialReference()
#     src.ImportFromEPSG(3413)
#     tgt.ImportFromEPSG(4326)
#     transform = osr.CoordinateTransformation(src, tgt)
#     itransform = osr.CoordinateTransformation(tgt, src)
#     x, y = geotrans[0], geotrans[3]
#     coords = transform.TransformPoint(x, y)
#     print(coords)
#     xn, yn, _ = itransform.TransformPoint(coords[0], coords[1] + 0.5, 0)

