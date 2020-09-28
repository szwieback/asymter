'''
Created on Aug 4, 2020

@author: simon
'''
from collections import namedtuple
from osgeo import gdal, gdalconst
import os
import numpy as np
import pickle
import zlib

Geospatial = namedtuple('Geospatial', ['shape', 'proj', 'geotrans'])
dt_np = {'float32': np.float32, 'uint8': np.uint8}
dt_gdal = {'float32': gdalconst.GDT_Float32, 'uint8': gdalconst.GDT_Byte}

def read_gdal(fntif):
    f = gdal.Open(fntif, gdal.GA_ReadOnly)
    im = f.ReadAsArray()
    proj = f.GetProjection()
    geotrans = f.GetGeoTransform()
    return im, proj, geotrans

def geospatial_from_file(fn):
    f = gdal.Open(fn, gdal.GA_ReadOnly)
    proj = f.GetProjection()
    geotrans = f.GetGeoTransform()
    shape = (f.RasterYSize, f.RasterXSize)
    geospatial = Geospatial(shape=shape, proj=proj, geotrans=geotrans)
    return geospatial

def resample_gdal(
        geospatial, inarr=None, inproj=None, ingeotrans=None, datatype='float32',
        src=None, average=False):
    import tempfile
    gs = geospatial
    with tempfile.TemporaryDirectory() as dirtmp:
        fntemp = os.path.join(dirtmp, 'res.tif')
        fntempsrc = os.path.join(dirtmp, 'src.tif')
        # Output / destination
        if src is None:
            n_bands = 1 if len(inarr.shape) == 2 else inarr.shape[0]
            inshape = inarr.shape[-2:]
            src = gdal.GetDriverByName('GTiff').Create(
                fntempsrc, inshape[1], inshape[0], n_bands, dt_gdal[datatype])
            src.SetGeoTransform(ingeotrans)
            src.SetProjection(inproj)
            for jb, arr_band in enumerate(np.reshape(inarr, (n_bands,) + inshape)):
                src.GetRasterBand(jb + 1).WriteArray(arr_band.astype(dt_np[datatype]))
        else:
            n_bands = src.RasterCount
        dst = gdal.GetDriverByName('GTiff').Create(
            fntemp, gs.shape[1], gs.shape[0], n_bands, dt_gdal[datatype])
        for jb in range(n_bands):
            a = np.ndarray(shape=gs.shape, dtype=dt_np[datatype])
            a.fill({'float32': np.nan, 'uint8': 255}[datatype])
            dst.GetRasterBand(jb + 1).WriteArray(a)
        dst.SetGeoTransform(gs.geotrans)
        dst.SetProjection(gs.proj)
        intmode = {'uint8': gdalconst.GRA_NearestNeighbour,
                   'float32': gdalconst.GRA_Bilinear}[datatype]
        if average: intmode = gdalconst.GRA_Average
        gdal.ReprojectImage(src, dst, inproj, gs.proj, intmode)
        resampled = dst.ReadAsArray()
        del dst
    return resampled

def index_from_coord(coords, geotransform):
    assert geotransform[2] == 0
    assert geotransform[4] == 0
    coordinatest = np.atleast_2d(coords)
    if coordinatest.shape[0] == 1:
        coordinatest = coordinatest.T
    ind = np.zeros_like(coordinatest, dtype=np.float64)
    ind[1, :] = (coordinatest[0, :] - geotransform[0]) / geotransform[1]
    ind[0, :] = (coordinatest[1, :] - geotransform[3]) / geotransform[5]
    return(ind)

def gdal_cclip(im, geotrans, cwindow):
    gt = geotrans
    cw = cwindow
    inda = index_from_coord((cw[0] + cw[2] / 2, cw[1] + cw[3] / 2), gt)[:, 0].astype(np.int64)
    indb = index_from_coord((cw[0] - cw[2] / 2, cw[1] - cw[3] / 2), gt)[:, 0].astype(np.int64)
    ll = (min(max(0, min(inda[0], indb[0])), im.shape[-2]),
          min(max(0, min(inda[1], indb[1])), im.shape[-1]))
    ur = (min(max(0, max(inda[0], indb[0])), im.shape[-2]),
          min(max(0, max(inda[1], indb[1])), im.shape[-1]))
    return im[..., ll[0]:ur[0], ll[1]:ur[1]]

def enforce_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

def save_geotiff(
        arr, fn, geotransform=None, proj=None, absolute=False, aggregate=None,
        datatype='float32', overwrite=True):
    enforce_directory(os.path.dirname(fn))
    def trans(vals):
        if absolute:
            return np.abs(vals)
        else:
            return vals
    def agg(vals):
        if aggregate == 'max':
            return np.max(vals, axis=0)
        else:
            return vals
    if not overwrite and os.path.exists(fn):
        return
    arr_ = agg(trans(arr))
    n_bands = 1 if len(arr_.shape) == 2 else arr.shape[0]
    dst = gdal.GetDriverByName('GTiff').Create(
        fn, arr_.shape[-1], arr_.shape[-2], n_bands, dt_gdal[datatype],
        ['COMPRESS=LZW', 'BIGTIFF=YES'])
    dst.SetGeoTransform(geotransform)
    dst.SetProjection(proj)
    if len(arr_.shape) == 2:
        dst.GetRasterBand(1).WriteArray(arr_.astype(dt_np[datatype]))
    else:
        for jb, arr_band in enumerate(arr_):
            dst.GetRasterBand(jb + 1).WriteArray(arr_band.astype(dt_np[datatype]))
    dst = None

def proj_from_epsg(code):
    from osgeo import osr
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(code)
    proj = sref.ExportToWkt()
    return proj

def save_object(obj, filename):
    enforce_directory(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    if os.path.splitext(filename)[1].strip() == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj
