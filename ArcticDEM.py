'''
Created on Aug 2, 2020

@author: simon
'''

import os
from osgeo import gdal, osr
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2

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
    shapeout = (np.array(arr.shape) * (1 + pct / 100)).astype(np.uint32)
    shapeout = tuple(shapeout + np.mod(shapeout, 2))
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

def _angle_meridian(geotrans):
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

def NS_slope_bp(dem, proj, geotrans, bp=(100, 2500), ang=None):
    # bp are e-folding scales
    assert proj == projdef
    speczpc = fftshift(fft2(zero_pad(dem, pct=25)))
    sfreq = spatial_frequencies(speczpc.shape, geotrans[1])
    spechpc, speclpc = np.ones_like(speczpc), np.ones_like(speczpc)
    if bp[1] is not None:
        spechpc -= _gaussian(sfreq, bp[1])
    if bp[0] is not None:
        speclpc *= _gaussian(sfreq, bp[0])
    if ang is None:
        ang = _angle_meridian(geotrans)
    specs = 2j * np.pi * (np.cos(ang) * sfreq[0][:, np.newaxis]
                        -np.sin(ang) * sfreq[1][np.newaxis, :])
    slopezp = np.real(ifft2(ifftshift(spechpc * speclpc * speczpc * specs)))
    slope = slopezp[0:dem.shape[0], 0:dem.shape[1]]
    return slope

def slope_discrete(dem, proj, geotrans, ang=None):
    if ang is None:
        ang = _angle_meridian(geotrans)
    nsloped1 = dem.copy()
    nsloped1[:-1, :] -= nsloped1[1:, :]
    nsloped1 /= geotrans[5]
    nsloped2 = dem.copy()
    nsloped2[:, :-1] -= nsloped2[:, 1:]
    nsloped2 /= geotrans[1]
    nsloped = np.cos(ang) * nsloped1 + np.sin(ang) * nsloped2
    return nsloped
    
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
    tile = (52, 19)#(49, 20)
    dem, proj, geotrans = read_tile(tile=tile)
    bp = (100, 1000)#3000 and larger have issues with large-scale slopes
    dem, mask_slope = inpaint_mask(dem)
    ang = _angle_meridian(geotrans)
#     ang = np.deg2rad(90)
#     print(np.rad2deg(ang))
#     dem = np.zeros((100, 100))
#     dem[:, :] = 100 * np.real(np.exp(1j * (np.cos(ang) * np.arange(dem.shape[0])[:, np.newaxis] - np.sin(ang) * np.arange(dem.shape[1])[np.newaxis, :]) / 20))
    nslope = NS_slope_bp(dem, proj, geotrans, bp=bp, ang=ang)
    nslope[mask_slope] = np.nan
    nslope
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(dem)
    vlim = 0.3
    ax[1].imshow(nslope, vmin=-vlim, vmax=vlim, cmap='bwr')
    nsloped = slope_discrete(dem, proj, geotrans, ang=ang)
    nsloped[mask_slope] = np.nan

    print(np.nanpercentile(nslope, 75), np.nanpercentile(nsloped, 75))

#     nslopew = nslope[1700:2400, 1250:2500]
    nslopew = nslope[1000:-1000, 1000:-1000]

    slopesum = np.nanpercentile(nslopew, (25, 50, 75))
    print(slopesum, np.nanmean(nslopew))
    print(100 * slopesum[1] / (slopesum[2] - slopesum[0]))

    ax[2].imshow(nsloped, vmin=-vlim, vmax=vlim, cmap='bwr')
    plt.show()

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

