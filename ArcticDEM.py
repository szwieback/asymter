'''
Created on Aug 2, 2020

@author: simon
'''

import os
from osgeo import gdal
import numpy as np

from paths import pathADEM
from IO import read_gdal, Geospatial, resample_gdal

ADEMversiondef = 'v3.0'
ADEMresdef = '32m'
ADEMinvalid = -9999.0,
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

def filename_ADEM(tile=(52, 19), path=pathADEM, res=ADEMresdef, version=ADEMversiondef):
    tilestr = f'{tile[0]}_{tile[1]}_{res}_{version}'
    fntif = os.path.join(path, res, tilestr, tilestr + '_reg_dem.tif')
    return fntif

def read_tile(tile=(52, 19), path=pathADEM, res=ADEMresdef, version=ADEMversiondef):
    fntif = filename_ADEM(tile=tile, path=path, res=res, version=version)
    dem, proj, geotrans = read_gdal(fntif)
    return dem, proj, geotrans

def tile_virtual_raster(
        tiles, fnvrt, path=pathADEM, res=ADEMresdef, version=ADEMversiondef):
    inputtif = [
        filename_ADEM(tile=tile, path=path, res=res, version=version) for tile in tiles]
    gdal.BuildVRT(fnvrt, [fntif for fntif in inputtif if os.path.exists(fntif)],
                  VRTNodata=np.nan)

def read_tile_buffer(
        tile=(52, 19), buffer=2e4, path=pathADEM, res=ADEMresdef, version=ADEMversiondef):
    from itertools import product
    dem_, proj, geotrans = read_tile(tile=tile, path=path, res=res, version=version)
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
        tile_virtual_raster(tiles, fnvrt, path=path, res=res, version=version)
        # resample
        src = gdal.Open(fnvrt, gdal.GA_ReadOnly)
        dem = resample_gdal(geospatial, datatype='float32', src=src)
    return dem, proj, geotrans_, bufferpix

def download_tile(
        tile=(50, 19), path=pathADEM, res=ADEMresdef, version=ADEMversiondef, 
        overwrite=False):
    import tarfile, requests
    tilestr = f'{tile[0]}_{tile[1]}_{res}_{version}'
    resurl = f'{version}/{res}/{tile[0]}_{tile[1]}/{tilestr}.tar.gz'
    fnlocal = os.path.join(path, res, f'{tilestr}.tar.gz')
    pathtile = os.path.join(path, res, tilestr)
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

def download_all_tiles(
        tilemax=74, path=pathADEM, res=ADEMresdef, version=ADEMversiondef, overwrite=False):
    from itertools import product
    tilenos = range(1, tilemax + 1)
    for tile in product(tilenos, tilenos):
        download_tile(tile=tile, path=path, res=res, version=version, overwrite=overwrite)
    


if __name__ == '__main__':
#     download_all_tiles()
    pass
