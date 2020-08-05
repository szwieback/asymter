'''
Created on Aug 4, 2020

@author: simon
'''
from collections import namedtuple
from osgeo import gdal, gdalconst
import os
import numpy as np

Geospatial = namedtuple('Geospatial', ['shape', 'proj', 'geotrans'])
dt_np = {'float32': np.float32, 'uint8': np.uint8}
dt_gdal = {'float32': gdalconst.GDT_Float32, 'uint8': gdalconst.GDT_Byte}

def read_gdal(fntif):
    f = gdal.Open(fntif, gdal.GA_ReadOnly)
    im = f.ReadAsArray()
    proj = f.GetProjection()
    geotrans = f.GetGeoTransform()
    return im, proj, geotrans

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