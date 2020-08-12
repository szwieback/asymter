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
                       ADEMinvalid, ADEMresdef, tilestr)

indices_bootstrap = ['median', 'logratio', 'roughness', 'medianEW', 'logratioEW']
seed = 1

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
    import warnings
    from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        dem[dem <= ADEMinvalid] = np.nan
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
        (-np.sin(ang) * sfreq[0][:, np.newaxis] - np.cos(ang) * sfreq[1][np.newaxis, :])),
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

def _median_asymindex(slope, indtype='median'):
    if indtype == 'median':
        indband = 0
    elif indtype == 'medianEW':
        indband = 1
    else:
        raise NotImplementedError(f'{indtype} not known')
    if slope.shape[1] > 50:
        slopep = np.percentile(slope[indband, ...], (25, 50, 75))
        asymi = -100 * slopep[1] / (slopep[2] - slopep[0])  # pos: N/E facing steeper
    else:
        asymi = np.nan
    return asymi

def _logratio_asymindex(slope, minslope=0.05, aspthresh=1, indtype='logratio', count=False):
    import warnings
    if indtype == 'logratioEW':
        slope_ = np.flip(slope, axis=0)
    elif indtype == 'logratio':
        slope_ = slope
    else:
        raise NotImplementedError(f'{indtype} not known')
    slope_abs = np.sqrt(np.sum(slope_ ** 2, axis=0))
    asp = slope_[0, ...] / np.abs(slope_[1, ...])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        slope_abs_0 = slope_abs[np.logical_and(asp >= aspthresh, slope_abs >= minslope)]
        slope_abs_1 = slope_abs[np.logical_and(asp <= -aspthresh, slope_abs >= minslope)]
        asymi = np.log10(np.median(slope_abs_0) / np.median(slope_abs_1))
    if count:
        asymi = min(len(slope_abs_0), len(slope_abs_1))
    return asymi

def _roughness(slope):
    if slope.shape[1] > 0:
        roughness = np.sqrt(np.mean(np.sum(slope ** 2, axis=0)))
    else:
        roughness = np.nan
    return roughness

def asymindex(slope, indtype='median', bootstrap_se=False, N_bootstrap=100, **kwargs):
    # slope: (2, N), no NaN
    if not bootstrap_se:
        # EW same sign convention as NS
        if indtype in ('median', 'medianEW'):
            asymi = _median_asymindex(slope, indtype=indtype)
        elif indtype in ('logratio', 'logratioEW'):
            asymi = _logratio_asymindex(slope, indtype=indtype, **kwargs)
        elif indtype == 'N':
            asymi = slope.shape[-1]
        elif indtype == 'N_logratio':
            asymi = _logratio_asymindex(slope, count=True, **kwargs)
        elif indtype == 'roughness':
            asymi = _roughness(slope)
        else:
            raise NotImplementedError(f'{indtype} not known')
    else:
        import warnings
        rng = np.random.default_rng(seed=seed)
        asymi_bs = []
        for _ in range(N_bootstrap):
            slope_bs = rng.choice(slope, size=slope.shape[1], axis=1)
            asymi_bs.append(asymindex(slope_bs, indtype=indtype, **kwargs))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            asymi = np.nanstd(asymi_bs, ddof=1)
    return asymi

def asymindex_pts(
        slope, pts, geotrans, cellsize=(25e3, 25e3), indtypes=['median'],
        bootstrap_se=False, N_bootstrap=100, valid=True, **kwargs):
    from collections import defaultdict
    asym = defaultdict(lambda: [])
    for pt in pts:
        cwindow = (pt[0], pt[1], cellsize[0], cellsize[1])
        if slope is not None:
            slopew = gdal_cclip(slope, geotrans, cwindow)
            slopew = np.reshape(slopew, (slopew.shape[0], -1))
            slopew = slopew[:, np.all(np.isfinite(slopew), axis=0)]
        for indtype in indtypes:
            if valid:
                asym[indtype].append(
                    asymindex(slopew, indtype=indtype, **kwargs))
                if bootstrap_se and indtype in indices_bootstrap:
                    asym[f'{indtype}_se'].append(
                        asymindex(slopew, indtype=indtype, bootstrap_se=True,
                                  N_bootstrap=N_bootstrap, **kwargs))
            else:
                asym[indtype].append(np.nan)
                if bootstrap_se and indtype in indices_bootstrap:
                    asym[f'{indtype}_se'].append(np.nan)

    return dict(asym)

def asymter_tile(
        pts, cellsize=(25e3, 25e3), tile=(53, 17), bp=(100, 2000), buffer_water=None,
        buffer_read=None, indtypes=['median'], water_cutoffpct=5.0, bootstrap_se=False,
        N_bootstrap=100, pathADEM=pathADEM, res=ADEMresdef, version=ADEMversiondef,
        **kwargs):
    if buffer_water is None:
        buffer_water = 2 * bp[0]
    if buffer_read is None:
        _bp = bp[1] if bp[1] is not None else bp[0]
        buffer_read = cellsize[0] + 10 * _bp

    slope = None
    valid = False
    geotrans = None
    if tile_available(tile=tile, path=pathADEM, res=res, version=version) and len(pts) > 0:
        dem, proj, geotrans, _ = read_tile_buffer(
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
            slope[:, mask] = np.nan
            valid = True
    asym = asymindex_pts(
        slope, pts, geotrans, cellsize=cellsize, indtypes=indtypes,
        bootstrap_se=bootstrap_se, N_bootstrap=N_bootstrap, valid=valid, **kwargs)
    return asym

def test_asymter():
    tile = (33, 60)  # (40, 17)  # (49, 20)
    bp = (100, None)  # 3000 and larger have issues with large-scale slopes
    buffer_water = 2 * bp[0]
    buffer_read = 1000  # 10 * bp[1]

    dem, proj, geotrans, bufferpix = read_tile_buffer(
        tile=tile, buffer=buffer_read, path=pathADEM, version=ADEMversiondef,
        res=ADEMresdef)

    ang = _angle_meridian(dem, proj, geotrans)
    print(ang, np.rad2deg(ang))
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
    indtypes=['median'], water_cutoffpct=5.0, bootstrap_se=False, N_bootstrap=100,
    overwrite=False, **kwargs):
    tile = tilestruc.tile
    fnout = os.path.join(pathout, f'{tilestr(tile)}.p')
    if not os.path.exists(fnout): print(fnout)
    if not os.path.exists(fnout) or overwrite:
        ptslist = list(tilestruc.pts)
        asymind_tile = asymter_tile(
            ptslist, cellsize=cellsize, tile=tile, bp=bp, indtypes=indtypes,
            water_cutoffpct=water_cutoffpct, bootstrap_se=bootstrap_se,
            N_bootstrap=N_bootstrap, **kwargs)
        dictout = {'tile': tilestruc.tile, 'asymind': asymind_tile, 'pts': ptslist,
                   'ind': list(tilestruc.ind)}
        save_object(dictout, fnout)
    else:
        dictout = load_object(fnout)
    return dictout

def _write_geotiff(pathout, scenname, grid, asyminds, geotrans, proj, indtype='median'):
    asymindarr = np.zeros((len(grid[1]), len(grid[0]))) + np.nan
    for asyminddict in asyminds:
        for jpt, ptind in enumerate(asyminddict['ind']):
            if len(ptind) >= 1:
                asymindarr[ptind[1], ptind[0]] = asyminddict['asymind'][indtype][jpt]
    fnout = os.path.join(pathout, f'{scenname}_{indtype}.tif')
    print(fnout)
    save_geotiff(asymindarr, fnout, geotransform=geotrans, proj=proj)

def batch_asymterr(
        scenname, indtypes=['median'], cellsize=(25e3, 25e3), bp=(100, 2000),
        water_cutoffpct=5.0, bootstrap_se=False, N_bootstrap=100, pathind=pathindices,
        overwrite=False, n_jobs=-1):
    from grids import gridtiles, create_grid, corner0, spacingdef, corner1, EPSGdef
    spacing = spacingdef
    grid = create_grid(spacing=spacing, corner0=corner0, corner1=corner1)
    kwargs = {}
    geotrans = (corner0[0], spacing[0], 0.0, corner0[1], 0.0, spacing[1])
    proj = proj_from_epsg(EPSGdef)

    pathout = os.path.join(pathind, scenname)
    enforce_directory(pathout)
    gt = gridtiles(grid)
    def _process(tilestruc):
        try:
            res = _batch_asymterr(
                tilestruc, pathout, cellsize=cellsize, bp=bp, indtypes=indtypes,
                water_cutoffpct=water_cutoffpct, bootstrap_se=bootstrap_se,
                N_bootstrap=N_bootstrap, overwrite=overwrite, **kwargs)
        except:
            print(f'Error in {tilestruc.tile}')
            res = None
        return res
    if n_jobs == 1:
        asyminds = []
        for tilestruc in gt:
            if tilestruc.tile == (53, 17):
                asyminds.append(_process(tilestruc))
    else:
        from joblib import Parallel, delayed
        asyminds = Parallel(n_jobs=n_jobs)(delayed(_process)(tilestruc) for tilestruc in gt)

    print('writing out geotiffs')
    indtypes_ = indtypes
    if bootstrap_se:
        indtypes_ = indtypes_ + [f'{it}_se' for it in indtypes if it in indices_bootstrap]
    for indtype in indtypes_: # add se
        _write_geotiff(pathout, scenname, grid, asyminds, geotrans, proj, indtype=indtype)

if __name__ == '__main__':
    indtypes = [
        'median', 'logratio', 'roughness', 'medianEW', 'logratioEW', 'N', 'N_logratio']
    batch_asymterr('bandpass', indtypes=indtypes, cellsize=(25e3, 25e3), bp=(100, 2000),
        water_cutoffpct=5.0, overwrite=False, bootstrap_se=True, N_bootstrap=25, n_jobs=-1)
    batch_asymterr('lowpass', indtypes=indtypes, cellsize=(25e3, 25e3), bp=(100, None),
        water_cutoffpct=5.0, overwrite=False, bootstrap_se=True, N_bootstrap=25, n_jobs=4)
#     test_asymter()
    # test region with large negative values