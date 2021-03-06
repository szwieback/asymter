'''
Created on Aug 6, 2020

@author: simon
'''
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import os

from asymter import (
    Geospatial, gdal_cclip, save_object, load_object, enforce_directory, save_geotiff,
    proj_from_epsg, path_adem, path_indices, match_watermask, read_adem_tile_buffer,
    adem_tile_available, adem_defversion, adem_definvalid, adem_defres, adem_tilestr)

indices_bootstrap_def = ['median', 'logratio', 'roughness', 'medianEW', 'logratioEW']
seed = 1
minslope_def = 0.04

def zero_pad(arr, pct=25):
    shapeout = (np.array(arr.shape) * (1 + pct / 100))
    shapeout = 2 ** (np.floor(np.log2(shapeout)) + 1)
    shapeout = tuple(shapeout.astype(np.uint32))
    arr_zp = np.zeros(shapeout, dtype=arr.dtype) + np.nanmean(arr)
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

def _aggressive_interpolation(dem, N = (5, 5)):
    from skimage.transform import resize
    import warnings
    dds = np.zeros(N)
    b = (np.array(dem.shape)/N).astype(np.int16)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for j0 in range(N[0]):
            for j1 in range(N[1]):
                dds[j0, j1] = np.nanmedian(dem[j0*b[0]:(j0+1)*b[0], j1*b[1]:(j1+1)*b[1]])
    dds[np.isnan(dds)] = np.nanmedian(dem)
    demfs = resize(dds, dem.shape)
    return demfs

def inpaint_mask(
        dem, geotrans, bp=(100, 1000), adem_definvalid=adem_definvalid, mf_factor=4):
    # to do: use opencv inpainting; reasonable length scales
    import warnings
    from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        dem[dem <= adem_definvalid] = np.nan
    demf = median_filter(dem, size=mf_factor * int(np.abs(bp[0] / geotrans[1])))#2
    dem[np.isnan(dem)] = demf[np.isnan(dem)]
    mask0 = np.isnan(dem)
    demfs = _aggressive_interpolation(dem)
    dem[mask0] = demfs[mask0]
    l_int = 2 * bp[1] if bp[1] is not None else 3 * bp[0] #2
    selem = generate_binary_structure(2, 1)
    mask = binary_dilation(
        mask0, selem, iterations=int(np.abs(l_int / geotrans[1])))
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

def _logratio_asymindex(
        slope, minslope=minslope_def, aspthresh=1.0, indtype='logratio', count=False):
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

def _ruggedness(dem):
    if dem.shape[1] > 0:
        ruggedness = np.percentile(dem[0, ...], 90) - np.percentile(dem[0, ...], 10)
    else:
        ruggedness = np.nan
    return ruggedness

def _absslope(slope, param='mean', minslope=minslope_def):
    if slope.shape[1] > 0:
        import warnings
        slope_abs = np.sqrt(np.sum(slope ** 2, axis=0))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            slope_abs = slope_abs[slope_abs >= minslope]
            if param == 'mean':
                absslope = np.nanmean(slope_abs)
            elif param == 'median':
                absslope = np.nanmedian(slope_abs)
            else:
                raise NotImplementedError
    else:
        absslope = np.nan
    return absslope
    
def block_bootstrap(topo, N_bootstrap=100, bs=(10, 10), rng=None):
    from itertools import product
    if rng is None:
        rng = np.random.RandomState(seed=1)
    # forms blocks over axes 1+
    N = np.floor(np.array(topo.shape)[1:] / np.array(bs)).astype(np.int64)
    N_tot = np.product(N)
    # discards a tiny fraction if topo shape not divisible by bs
    topoblock = np.zeros((topo.shape[0], *bs, N_tot))
    # loop: b is a tuple of block indices
    for jb, b in enumerate(product(*[range(x) for x in N])):
        s = [slice(None)] + [slice(b_ * bs_, (b_ + 1) * bs_) for b_, bs_ in zip(b, bs)]
        topoblock[..., jb] = topo[tuple(s)]
    for _ in range(N_bootstrap):
        block_sample = rng.choice(range(N_tot), size=N_tot)
        topo_sample = np.concatenate(
            [topoblock[..., b].reshape(topo.shape[0], -1) for b in block_sample], axis=-1)
        # remove nan
        yield topo_sample[:, np.all(np.isfinite(topo_sample), axis=0)]

def asymindex(topow, indtype='median', bootstrap_se=False, N_bootstrap=100, **kwargs):
    # topow: (2, ...) [slope: NS, EW], or (1, ...) [just DEM] if noslope
    # EW same sign convention as NS
    if not bootstrap_se:
        topo = np.reshape(topow, (topow.shape[0], -1))
        topo = topo[:, np.all(np.isfinite(topo), axis=0)]
        parts = indtype.split('_')
        if indtype in ('median', 'medianEW'):
            asymi = _median_asymindex(topo, indtype=indtype)
        elif indtype in ('logratio', 'logratioEW'):
            asymi = _logratio_asymindex(topo, indtype=indtype, **kwargs)
        elif indtype == 'N':
            asymi = topo.shape[-1]
        elif indtype == 'N_logratio':
            asymi = _logratio_asymindex(topo, count=True, **kwargs)
        elif indtype == 'roughness':
            asymi = _roughness(topo)
        elif indtype == 'ruggedness':
            asymi = _ruggedness(topo)
        elif parts[0] == 'absslope':
            asymi = _absslope(topo, param=parts[1], **kwargs)
        else:
            raise NotImplementedError(f'{indtype} not known')
    else:
        import warnings
        rng = np.random.default_rng(seed=seed)
        asymi_bs = []
        for topo_bs in block_bootstrap(topow, N_bootstrap=N_bootstrap, rng=rng):
#             topo_bs = rng.choice(topoblock, size=topoblock.shape[1], axis=1)
            asymi_bs.append(asymindex(topo_bs, indtype=indtype, **kwargs))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            asymi = np.nanstd(asymi_bs, ddof=1)
    return asymi

def asymindex_pts(
        topo, pts, geotrans, cellsize=(25e3, 25e3), indtypes=['median'],
        bootstrap_se=False, indices_bootstrap=None, N_bootstrap=100, valid=True, **kwargs):
    from collections import defaultdict
    asym = defaultdict(lambda: [])
    if indices_bootstrap is None: indices_bootstrap = indices_bootstrap_def
    for pt in pts:
        cwindow = (pt[0], pt[1], cellsize[0], cellsize[1])
        if topo is not None:
            topow = gdal_cclip(topo, geotrans, cwindow)
        for indtype in indtypes:
            if valid:
                asym[indtype].append(
                    asymindex(topow, indtype=indtype, **kwargs))
                if bootstrap_se and indtype in indices_bootstrap:
                    asym[f'{indtype}_se'].append(
                        asymindex(topow, indtype=indtype, bootstrap_se=True,
                                  N_bootstrap=N_bootstrap, **kwargs))
            else:
                asym[indtype].append(np.nan)
                if bootstrap_se and indtype in indices_bootstrap:
                    asym[f'{indtype}_se'].append(np.nan)

    return dict(asym)

def asymter_tile(
        pts, cellsize=(25e3, 25e3), tile=(53, 17), bp=(100, 2000), buffer_water=None,
        buffer_read=None, indtypes=['median'], water_cutoffpct=5.0, bootstrap_se=False,
        indices_bootstrap=None, N_bootstrap=100, noslope=False, path_adem=path_adem, 
        res=adem_defres, version=adem_defversion, **kwargs):
    if buffer_water is None:
        buffer_water = 2 * bp[0]
    if buffer_read is None:
        _bp = bp[1] if bp[1] is not None else bp[0]
        buffer_read = cellsize[0] + 10 * _bp

    topo = None
    valid = False
    geotrans = None
    if (adem_tile_available(tile=tile, path=path_adem, res=res, version=version)
        and len(pts) > 0):
        dem, proj, geotrans, _ = read_adem_tile_buffer(
            tile=tile, buffer=buffer_read, path=path_adem, version=adem_defversion,
            res=adem_defres)
        # azimuth angle of meridian
        ang = _angle_meridian(dem, proj, geotrans)
        # masks
        dem, mask_gap = inpaint_mask(dem, geotrans, bp=bp)
        geospatial = Geospatial(shape=dem.shape, proj=proj, geotrans=geotrans)
        mask_water = match_watermask(
            geospatial, cutoffpct=water_cutoffpct, buffer=buffer_water)
        mask = np.logical_or(mask_water, mask_gap)
        valid = True
        if np.count_nonzero(np.isfinite(mask)) > 0 and not noslope:
            topo = slope_bp(dem, proj, geotrans, bp=bp, ang=ang)
            topo[:, mask] = np.nan
        elif np.count_nonzero(np.isfinite(mask)) and noslope:
            topo = dem[np.newaxis, ...]
            topo[:, mask] = np.nan
        else:
            valid = False
    asym = asymindex_pts(
        topo, pts, geotrans, cellsize=cellsize, indtypes=indtypes,
        bootstrap_se=bootstrap_se, indices_bootstrap=indices_bootstrap, 
        N_bootstrap=N_bootstrap, valid=valid, **kwargs)
    return asym

def _batch_asymter(
        tilestruc, pathout, cellsize=(25e3, 25e3), bp=(100, 2000),
        indtypes=['median'], water_cutoffpct=5.0, bootstrap_se=False, 
        indices_bootstrap=None, N_bootstrap=100, noslope=False, overwrite=False, **kwargs):
    tile = tilestruc.tile
    fnout = os.path.join(pathout, f'{adem_tilestr(tile)}.p')
    if not os.path.exists(fnout): print(fnout)
    if not os.path.exists(fnout) or overwrite:
        ptslist = list(tilestruc.pts)
        asymind_tile = asymter_tile(
            ptslist, cellsize=cellsize, tile=tile, bp=bp, indtypes=indtypes,
            water_cutoffpct=water_cutoffpct, bootstrap_se=bootstrap_se,
            indices_bootstrap=indices_bootstrap, N_bootstrap=N_bootstrap, noslope=noslope, 
            **kwargs)
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
    save_geotiff(asymindarr, fnout, geotransform=geotrans, proj=proj)

def batch_asymter(
        scenname, indtypes=['median'], cellsize=(25e3, 25e3), bp=(100, 2000),
        spacing=None, water_cutoffpct=25.0, bootstrap_se=False, indices_bootstrap=None, 
        N_bootstrap=100, pathind=path_indices, noslope=False, overwrite=False, n_jobs=-1, 
        **kwargs):
    from asymter import gridtiles, create_grid, corner0, spacingdef, corner1, EPSGdef
    if spacing is None:
        spacing = spacingdef
    if indices_bootstrap is None: indices_bootstrap = indices_bootstrap_def
    grid = create_grid(spacing=spacing, corner0=corner0, corner1=corner1)
    geotrans = (corner0[0], spacing[0], 0.0, corner0[1], 0.0, spacing[1])
    proj = proj_from_epsg(EPSGdef)
    pathout = os.path.join(pathind, scenname)
    enforce_directory(pathout)
    gt = gridtiles(grid)
    def _process(tilestruc):
        try:
            res = _batch_asymter(
                tilestruc, pathout, cellsize=cellsize, bp=bp, indtypes=indtypes,
                water_cutoffpct=water_cutoffpct, bootstrap_se=bootstrap_se,
                indices_bootstrap=indices_bootstrap, N_bootstrap=N_bootstrap, 
                noslope=noslope, overwrite=overwrite, **kwargs)
        except:
            print(f'error in {tilestruc.tile}')
#             raise
            res = None
        return res
    if n_jobs == 1:
        asyminds = []
        for tilestruc in gt:
            asyminds.append(_process(tilestruc))
    else:
        from joblib import Parallel, delayed
        asyminds = Parallel(n_jobs=n_jobs)(delayed(_process)(tilestruc) for tilestruc in gt)

    indtypes_ = indtypes
    if bootstrap_se:
        indtypes_ = indtypes_ + [f'{it}_se' for it in indtypes if it in indices_bootstrap]
    for indtype in indtypes_:  # add se
        _write_geotiff(pathout, scenname, grid, asyminds, geotrans, proj, indtype=indtype)

