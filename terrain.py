'''
Created on Aug 6, 2020

@author: simon
'''
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import os

from paths import pathADEM, pathindices
from IO import (Geospatial, gdal_cclip, save_object, load_object, enforce_directory,
                save_geotiff, proj_from_epsg)
from watermask import match_watermask
from ArcticDEM import (read_tile_buffer, tile_available, ADEMversiondef,
                       ADEMinvalid, ADEMresdef)

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

def _angle_meridian(dem, proj, geotrans):
    assert 'Polar_Stereographic' in proj
#     assert proj == projdef
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

def bandpass(dem, geotrans, bp=(None, None), zppct=25.0, freqdomain=False):
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

def inpaint_mask(dem, geotrans, bp=(100, 1000), ADEMinvalid=ADEMinvalid):
    # to do: use opencv inpainting; reasonable length scales
    dem[dem <= ADEMinvalid] = np.nan
    from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure
    demf = median_filter(dem, size=2 * int(np.abs(bp[0] / geotrans[1])))
    dem[np.isnan(dem)] = demf[np.isnan(dem)]
    mask = np.isnan(dem)
    l_int = 2 * bp[1] if bp[1] is not None else 3 * bp[0]
    selem = generate_binary_structure(2, 1)
    mask = binary_dilation(
        mask, selem, iterations=int(np.abs(l_int / geotrans[1])))
    dem[mask] = np.nanmedian(dem)
    return dem, mask

def slope_bp(dem, proj, geotrans, bp=(100, 2500), ang=None, zppct=25.0):
    if ang is None:
        ang = _angle_meridian(dem, proj, geotrans)
    speczpc, sfreq = bandpass(dem, geotrans, bp=bp, zppct=zppct, freqdomain=True)
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
        ang = _angle_meridian(dem, proj, geotrans)
    if bp != (None, None):
        dem_ = bandpass(dem, geotrans, bp=bp, zppct=zppct)
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
    import warnings
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

def asymindex_pts(slope, pts, geotrans, cellsize=(25e3, 25e3), indtype='median', **kwargs):
    asymind_pts = []
    for pt in pts:
        cwindow = (pt[0], pt[1], cellsize[0], cellsize[1])
        slopew = gdal_cclip(slope, geotrans, cwindow)
        asymind_pts.append(asymindex(slopew, indtype=indtype, **kwargs))
    return asymind_pts

def asymter_tile(
        pts, cellsize=(25e3, 25e3), tile=(53, 17), bp=(100, 2000), buffer_water=None,
        buffer_read=None, indtype='median', pathADEM=pathADEM, res=ADEMresdef,
        version=ADEMversiondef, water_cutoffpct=5.0, **kwargs):
    if buffer_water is None:
        buffer_water = 2 * bp[0]
    if buffer_read is None:
        buffer_read = cellsize[0] + 10 * bp[1]

    if tile_available(tile=tile, path=pathADEM, res=res, version=version) and len(pts) > 0:
        dem, proj, geotrans, bufferpix = read_tile_buffer(
            tile=tile, buffer=buffer_read, path=pathADEM, version=ADEMversiondef,
            res=ADEMresdef)
        # azimuth angle of meridian
        ang = _angle_meridian(dem, proj, geotrans)
        # masks
        dem, mask_gap = inpaint_mask(dem, geotrans, bp=bp)
        geospatial = Geospatial(shape=dem.shape, proj=proj, geotrans=geotrans)
        mask_water = match_watermask(
            geospatial, cutoffpct=water_cutoffpct, buffer=buffer_water)
        mask = np.logical_or(mask_water, mask_gap)
        if np.count_nonzero(np.isfinite(mask)) > 0:
            slope = slope_bp(dem, proj, geotrans, bp=bp, ang=ang)
        else:
            slope = np.stack((dem, dem), axis=0)
        slope[:, mask] = np.nan

        asym = asymindex_pts(
            slope, pts, geotrans, cellsize=cellsize, indtype=indtype, **kwargs)
    else:
        asym = list(np.ones((len(pts),)) * np.nan)
    return asym

def test_asymter():
    tile = (34, 42)  # (40, 17)  # (49, 20)
    bp = (100, 2000)  # 3000 and larger have issues with large-scale slopes
    buffer_water = 2 * bp[0]
    buffer_read = 10 * bp[1]

    dem, proj, geotrans, bufferpix = read_tile_buffer(
        tile=tile, buffer=buffer_read, path=pathADEM, version=ADEMversiondef,
        res=ADEMresdef)

    print(geotrans, np.array(dem.shape) * np.array((geotrans[1], geotrans[5])))
    ang = _angle_meridian(dem, proj, geotrans)

#     ang = np.deg2rad(90)
#     dem = np.zeros((4096, 4096))
#     dem[:, :] = 100 * np.real(np.exp(1j * (np.cos(ang) * np.arange(dem.shape[0])[:, np.newaxis] - np.sin(ang) * np.arange(dem.shape[1])[np.newaxis, :]) / 20))

    dem, mask_gap = inpaint_mask(dem, geotrans, bp=bp)
    geospatial = Geospatial(shape=dem.shape, proj=proj, geotrans=geotrans)
    mask_water = match_watermask(geospatial, cutoffpct=5.0, buffer=buffer_water)
    mask = np.logical_or(mask_water, mask_gap)

    slope = slope_bp(dem, proj, geotrans, bp=bp, ang=ang)
    slope[:, mask] = np.nan

    sloped = slope_discrete_bp(dem, proj, geotrans, ang=ang, bp=bp)
    sloped[:, mask] = np.nan

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=3)
#     dem = bandpass(dem, geotrans, bp=bp)
    ax[0].imshow(dem)
    vlim = 0.3
    ax[1].imshow(slope[0, ...], vmin=-vlim, vmax=vlim, cmap='bwr')
    ax[2].imshow(sloped[0, ...], vmin=-vlim, vmax=vlim, cmap='bwr')

#     nslopew = nslope[1700:2400, 1250:2500]
    for slope_ in (slope, sloped):
#         slopew = slope_[:, 1000:-1000, 1000:-1000]
        slopew = slope_[:, bufferpix[0]:-bufferpix[0], bufferpix[1]:-bufferpix[1]]
        print(bufferpix, slopew.shape)
        print(asymindex(slopew), asymindex(slopew, indtype='logratio'))
        print('     summary', np.nanmean(slopew[0, ...]),
              np.nanpercentile(slopew[0, ...], (25, 50, 75)))  # also store the mean/median
    # do scatterplot comparison
    plt.show()

def _batch_asymterr(tilestruc, pathout, cellsize=(25e3, 25e3), bp=(100, 2000),
    indtype='median', water_cutoffpct=5.0, overwrite=False, **kwargs):
    tile = tilestruc.tile
    fnout = os.path.join(pathout, f'{tile[0]}_{tile[1]}.p')
    if not os.path.exists(fnout) or overwrite:
        ptslist = list(tilestruc.pts)
        asymind_tile = asymter_tile(
            ptslist, cellsize=cellsize, tile=tile, bp=bp, indtype=indtype,
            water_cutoffpct=water_cutoffpct, **kwargs)
        dictout = {'tile': tilestruc.tile, 'asymind': asymind_tile, 'pts': ptslist,
                   'ind': list(tilestruc.ind)}
        save_object(dictout, fnout)
    else:
        dictout = load_object(fnout)
    return dictout

def batch_asymterr(
        scenname, indtype='median', pathind=pathindices, overwrite=False, n_jobs=-1):
    from grids import gridtiles, create_grid, corner0, spacingdef, corner1, EPSGdef
    spacing = spacingdef
    grid = create_grid(spacing=spacing, corner0=corner0, corner1=corner1)
    kwargs = {}
    cellsize = (25e3, 25e3)
    bp = (100, 2000)
    water_cutoffpct = 5.0
    geotrans = (corner0[0], spacing[0], 0.0, corner0[1], 0.0, spacing[1])
    proj = proj_from_epsg(EPSGdef)
    
    pathout = os.path.join(pathind, scenname)
    enforce_directory(pathout)
    gt = gridtiles(grid)
    def _process(tilestruc):
        res = _batch_asymterr(
            tilestruc, pathout, cellsize=cellsize, bp=bp, indtype=indtype,
            water_cutoffpct=water_cutoffpct, overwrite=overwrite, **kwargs)
        return res
    if n_jobs == 1:
        asyminds = []
        for tilestruc in gt:
            asyminds.append(_process(tilestruc))
    else:
        from joblib import Parallel, delayed
        asymind = Parallel(n_jobs=n_jobs)(delayed(_process)(tilestruc) for tilestruc in gt)
    asymindarr = np.zeros((len(grid[1]), len(grid[0]))) + np.nan
    for asyminddict in asymind:
        for ptind, asymind in zip(asyminddict['ind'], asyminddict['asymind']):
            asymindarr[ptind[1], ptind[0]] = asymind
    fnout = os.path.join(pathout, f'{scenname}.tif')
    save_geotiff(asymindarr, fnout, geotransform=geotrans, proj=proj)

if __name__ == '__main__':
#     test_asymter()
#     batch_asymterr('medianstd', indtype='median', overwrite=False)
    batch_asymterr('logratiostd', indtype='logratio', overwrite=False)

    # fix geolocation error
    # test region with large negative values
