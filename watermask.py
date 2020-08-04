'''
Created on Aug 2, 2020

@author: simon
'''

import requests
import os
import itertools

pathwm = '/home/simon/Work/asymter/watermask'
patterndef = ('occurrence', 'occurrence_{0}_{1}v1_1_2019.tif')
url0 = 'https://storage.googleapis.com/global-surface-water/downloads2019v2'

def download_single(
        tile=('160W', '70N'), pathlocal=pathwm, pattern=patterndef, overwrite=False):
    fn = pattern[1].format(*tile)
    fnlocal = os.path.join(pathlocal, fn)
    if overwrite or not os.path.exists(fnlocal):
        url = f'{url0}/{pattern[0]}/{fn}'
        response = requests.get(url)
        with open(fnlocal, 'wb') as f:
            f.write(response.content)
                    
def download_Arctic(pathlocal=pathwm, pattern=patterndef, overwrite=False):
    t0 = [f'{lon0}{di}' for lon0 in range(0, 180, 10) for di in ('E', 'W')]
    t1 = ['70N', '80N']
    for tile in itertools.product(t0, t1):
        try:
            assert tile[0] != '0W'
            download_single(
                tile=tile, pathlocal=pathlocal, pattern=pattern, overwrite=overwrite)
        except:
            print(f'could not download {tile}')
    
if __name__ == '__main__':
    download_Arctic()
    