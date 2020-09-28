'''
Created on Aug 2, 2020

@author: simon
'''

import os
from osgeo import gdal
import numpy as np

from asymter import path_adem, read_gdal, Geospatial, resample_gdal

adem_defversion = 'v3.0'
adem_defres = '32m'
adem_definvalid = -9999.0,
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
absurl = 'http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/'

def adem_tilestr(tile):
    return str(tile[0]).zfill(2) + '_' + str(tile[1]).zfill(2)

def adem_filename(tile=(52, 19), path=path_adem, res=adem_defres, version=adem_defversion):
    tilename = f'{adem_tilestr(tile)}_{res}_{version}'
    fntif = os.path.join(path, res, tilename, tilename + '_reg_dem.tif')
    return fntif

def read_adem_tile(tile=(52, 19), path=path_adem, res=adem_defres, version=adem_defversion):
    fntif = adem_filename(tile=tile, path=path, res=res, version=version)
    dem, proj, geotrans = read_gdal(fntif)
    return dem, proj, geotrans

def adem_tile_available(
        tile=(52, 19), path=path_adem, res=adem_defres, version=adem_defversion):
    return os.path.exists(adem_filename(tile=tile, path=path, res=res, version=version))

def adem_tile_virtual_raster(
        tiles, fnvrt, path=path_adem, res=adem_defres, version=adem_defversion):
    inputtif = [
        adem_filename(tile=tile, path=path, res=res, version=version) for tile in tiles]
    gdal.BuildVRT(fnvrt, [fntif for fntif in inputtif if os.path.exists(fntif)],
                  VRTNodata=np.nan)

def read_adem_tile_buffer(
        tile=(52, 19), buffer=2e4, path=path_adem, res=adem_defres, version=adem_defversion):
    from itertools import product
    dem_, proj, geotrans = read_adem_tile(tile=tile, path=path, res=res, version=version)
    # determine coords
    assert geotrans[2] == geotrans[4] == 0
    bufferpix = (abs(int(buffer / geotrans[1])), abs(int(buffer / geotrans[5])))
    pos0 = (geotrans[0] - np.sign(geotrans[1]) * buffer,
            geotrans[3] - np.sign(geotrans[5]) * buffer)
    # need to convert to non-numpy int
    shape_ = tuple(int(l) for l in dem_.shape + np.abs(np.array(bufferpix)) * 2)
    geotrans_ = (pos0[0], geotrans[1], 0.0, pos0[1], 0.0, geotrans[5])
    geospatial = Geospatial(shape=shape_, proj=proj, geotrans=geotrans_)
    tiles = list(
        product([tile[0] + d for d in [-1, 0, 1]], [tile[1] + d for d in [-1, 0, 1]]))
    # create virtual raster
    import tempfile
    with tempfile.TemporaryDirectory() as dirtmp:
        fnvrt = os.path.join(dirtmp, 'merged.vrt')
        adem_tile_virtual_raster(tiles, fnvrt, path=path, res=res, version=version)
        # resample
        src = gdal.Open(fnvrt, gdal.GA_ReadOnly)
        dem = resample_gdal(geospatial, datatype='float32', src=src)
    return dem, proj, geotrans_, bufferpix

def download_adem_tile(
        tile=(50, 19), path=path_adem, res=adem_defres, version=adem_defversion, 
        overwrite=False):
    import tarfile, requests
    tilename = f'{adem_tilestr(tile)}_{res}_{version}'
    resurl = f'{version}/{res}/{adem_tilestr(tile)}/{tilename}.tar.gz'
    fnlocal = os.path.join(path, res, f'{tilename}.tar.gz')
    pathtile = os.path.join(path, res, tilename)
    if overwrite or not os.path.exists(pathtile):
        url = absurl + resurl
        response = requests.get(url)
        print(url, response.status_code)
        if response.status_code != 404:
            with open(fnlocal, 'wb') as f:
                print(f'downloading {url} to {fnlocal}')
                f.write(response.content)
            try:
                with tarfile.open(fnlocal, 'r:gz') as tar:
                    tar.extractall(pathtile)
                    if os.path.exists(fnlocal):
                        os.remove(fnlocal)
            except:
                print(f'Could not download {url}')

def download_all_adem_tiles(
        tilemax=74, path=path_adem, res=adem_defres, version=adem_defversion, overwrite=False):
    from itertools import product
    tilenos = range(1, tilemax + 1)
    for tile in product(tilenos, tilenos):
        download_adem_tile(
            tile=tile, path=path, res=res, version=version, overwrite=overwrite)
    
if __name__ == '__main__':
    download_all_adem_tiles()
    pass