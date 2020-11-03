'''
Created on Nov 3, 2020

@author: simon
'''
import os

from asymter import path0, path_indices

path_explanatory_ = os.path.join(path0, 'explanatory')
path_explanatory = {'soil': os.path.join(path_explanatory_, 'soil'),
                    'resampled': os.path.join(path_explanatory_, 'resampled'),
                    'glacier': os.path.join(path_explanatory_, 'glacier'),
                    'wind': os.path.join(path_explanatory_, 'wind')}
fnexplandict = {'ruggedness': os.path.join(path_indices, 'raw', 'raw_ruggedness.tif'),
                'soil': os.path.join(path_explanatory['resampled'], 'soil.tif'),
                'prec': os.path.join(path_explanatory['resampled'], 'prec.tif'),
                'temp': os.path.join(path_explanatory['resampled'], 'temp10.tif'),
                'wind': os.path.join(path_explanatory['resampled'], 'wind.tif'),
                'glacier': os.path.join(
                    path_explanatory['resampled'], 'glacier_simp.gpkg')}