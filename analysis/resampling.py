'''
Created on Oct 19, 2020

@author: simon
'''
import os
import numpy as np
from collections import defaultdict
import ogr
import osr

from setup import setup_path
setup_path()
from asymter import (
    path_explanatory, path_indices, resample_gdal, geospatial_from_file,
    save_geotiff, read_gdal)

fnref = os.path.join(path_indices, 'raw', 'raw_ruggedness.tif')
geospatial = geospatial_from_file(fnref)

modays = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

def rasterize(fnpolygon, geospatial, attribute=None):
    import gdal, gdalconst
    import tempfile
    dt_gdal = {'float32': gdalconst.GDT_Float32, 'uint8': gdalconst.GDT_Byte}
    gs = geospatial
    with tempfile.TemporaryDirectory() as dirtmp:
        fntmp = os.path.join(dirtmp, 'res.tif')
        vec = ogr.Open(fnpolygon, 0)  # read only
        lyr = vec.GetLayer()
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(fntmp, gs.shape[1], gs.shape[0], 1, dt_gdal['uint8'])
        dst_ds.SetGeoTransform(gs.geotrans)
        dst_ds.SetProjection(gs.proj)
        if attribute is None:
            gdal.RasterizeLayer(dst_ds, [1], lyr, None)
        else:
            opt = ['ATTRIBUTE=' + attribute]
            gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=opt)
        rpolygon = dst_ds.ReadAsArray()
    return rpolygon

def resample_soil(geospatial=geospatial, fnout=None):
    fnsoil = os.path.join(
        path_explanatory['soil'], 'average_soil_and_sedimentary-deposit_thickness.tif')
    raw, inproj, ingeotrans = read_gdal(fnsoil)
    invalid = raw == -1
    raw = raw.astype(np.float32)
    raw[invalid] = np.nan
    soil = resample_gdal(
        geospatial, inarr=raw, inproj=inproj, ingeotrans=ingeotrans, datatype='float32',
        average=True)
    if fnout is not None:
        save_geotiff(soil, fnout, proj=geospatial.proj, geotransform=geospatial.geotrans)
    return soil

def resample_climate(pathin, prod='temp10', fnout=None, thresh=-1000):
    if prod == 'temp10':
        fun_fnchelsa = lambda month: f'CHELSA_{prod}_{month}_1979-2013_V1.2_land.tif'
    else:
        fun_fnchelsa = lambda month: f'CHELSA_{prod}_{month}_V1.2_land.tif'
    fac = {'temp10': 0.1, 'prec': 1.0}[prod]
    total = {'temp10': False, 'prec': True}[prod]
    days_sum = np.sum(modays)

    for days, month in zip(modays, range(1, 13)):
        if not total:
            weight = fac * days / days_sum # mean
        else:
            weight = fac
        fnin = os.path.join(pathin, fun_fnchelsa(month))
        print(fnin)
        im, inproj, ingeotrans = read_gdal(fnin)
        im = im.astype(np.float32)
        im[im < thresh] = np.nan
        if month == 1:
            imw = im * weight
        else:
            imw += im * weight
    del im
    imw = resample_gdal(
        geospatial, inarr=imw, inproj=inproj, ingeotrans=ingeotrans, datatype='float32',
        average=True)
    if fnout is not None:
        save_geotiff(imw, fnout, proj=geospatial.proj, geotransform=geospatial.geotrans)
    return imw

def max_glacial_extent(fnout=None, fnoutvec=None):
    periods = ['30 ka', '35 ka', '40 ka', '45 ka', 'MIS 4', 'MIS 5a', 'MIS 5b', 
               'MIS 5c', 'MIS 5d', 'MIS 6', 'MIS 8', 'MIS 10', 'MIS 12', 'MIS 16', 
               'MIS 20-24']
    import gdal
    path0 = path_explanatory['glacier']
    def filename(period):
        folder_recons = defaultdict(lambda: 'hypothesised ice-sheet reconstructions')
        folder_recons['LGM'] = 'hypothesised ice-sheet reconstruction'
        path1 = os.path.join(path0, period, folder_recons[period])
        nameshp = [fn for fn in os.listdir(path1) if fn.endswith('_best_estimate.shp')][0]
        return os.path.join(os.path.join(path1, nameshp))
    rextent = np.zeros(geospatial.shape, dtype=np.bool)
    for period in periods:
        rperiod = rasterize(filename(period), geospatial=geospatial).astype(np.bool)
        rextent = np.logical_or(rextent, rperiod)
    if fnout is not None:
        save_geotiff(
            rextent.astype(np.uint8), fnout, geotransform=geospatial.geotrans, 
            proj=geospatial.proj, datatype='uint8')
        if fnoutvec is not None:
            rds = gdal.Open(fnout, gdal.GA_ReadOnly)
            band = rds.GetRasterBand(1)
            vds = ogr.GetDriverByName('GPKG').CreateDataSource(fnoutvec)
            vlayer = vds.CreateLayer('polygon', srs=osr.SpatialReference(wkt=geospatial.proj))
            fd = ogr.FieldDefn('value', ogr.OFTInteger)
            vlayer.CreateField(fd)
            field = vlayer.GetLayerDefn().GetFieldIndex('dn')
            gdal.Polygonize(band, band, vlayer, field, [], callback=None)

if __name__ == '__main__':
    fnsoil = os.path.join(path_explanatory['resampled'], 'soil.tif')
#     resample_soil(fnout=fnsoil)

#     pathin = '/media/simon/Seagate Backup Plus Drive/climate'
#     for prod in ['prec', 'temp10']:
#         fnprod = os.path.join(path_explanatory['resampled'], f'{prod}.tif')
#         resample_climate(pathin, prod=prod, fnout=fnprod)
    fnglacier = os.path.join(path_explanatory['resampled'], 'glacier.tif')
    fnglaciervec = os.path.join(path_explanatory['resampled'], 'glacier.gpkg')
    max_glacial_extent(fnout=fnglacier, fnoutvec=fnglaciervec)
