'''
Created on Aug 2, 2020

@author: simon
'''

import requests
import os
import itertools
from osgeo import gdal
import numpy as np
from urllib.parse import urljoin

from IO import resample_gdal, read_gdal

pathwm = '/home/simon/Work/asymter/watermask'
pathwmmerged = os.path.join(pathwm, 'merged')
patterndef = ('occurrence', 'occurrence_{0}_{1}v1_1_2019.tif')
url0 = 'https://storage.googleapis.com/global-surface-water/downloads2019v2'
if not os.path.exists(pathwmmerged): os.makedirs(pathwmmerged)
fnvrt = os.path.join(pathwmmerged, 'merged.vrt')

def download_single(
        tile=('160W', '70N'), pathlocal=pathwm, pattern=patterndef, overwrite=False):
    fn = pattern[1].format(*tile)
    fnlocal = os.path.join(pathlocal, fn)
    if overwrite or not os.path.exists(fnlocal):
        url = urljoin(url0, f'{pattern[0]}/{fn}')
        response = requests.get(url)
        with open(fnlocal, 'wb') as f:
            f.write(response.content)

def download_Arctic(pathlocal=pathwm, pattern=patterndef, overwrite=False):
    t0 = [f'{lon0}{di}' for lon0 in range(0, 190, 10) for di in ('E', 'W')]
    t1 = ['70N', '80N']
    for tile in itertools.product(t0, t1):
        try:
            assert tile[0] not in ('0W', '180E')
            download_single(
                tile=tile, pathlocal=pathlocal, pattern=pattern, overwrite=overwrite)
        except:
            print(f'could not download {tile}')

def panArctic_virtual(pattern=patterndef, fnvrt=fnvrt, pathwm=pathwm):
    import glob
    inputtif = glob.glob(os.path.join(pathwm, pattern[1].format('*', '*')))
    gdal.BuildVRT(fnvrt, inputtif)

def match_watermask(geospatial, fnvrt=fnvrt, cutoffpct=5.0, buffer=100):
    src = gdal.Open(fnvrt, gdal.GA_ReadOnly)
    wm = resample_gdal(geospatial, datatype='uint8', src=src)
    wm = np.logical_not(wm < int(cutoffpct))
    from scipy.ndimage import binary_dilation, generate_binary_structure
    if buffer is not None:
        selem = generate_binary_structure(2, 1)
        wm = binary_dilation(
            wm, selem, iterations=int(np.abs(buffer / geospatial.geotrans[1])))
    return wm

if __name__ == '__main__':
#     download_Arctic()
#     panArctic_virtual()
    pass

