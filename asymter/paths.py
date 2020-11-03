'''
Created on Aug 5, 2020

@author: simon
'''
import os
import socket

hostname = socket.gethostname()   
if hostname == 'asf-simon':
    path0 = '/home/simon/Work/asymter'
elif hostname == 'Vienna':
    path0 = '/10TBstorage/Work/asymter'
else:
    path0 = os.path.expanduser('~')

path_adem = os.path.join(path0, 'ArcticDEM')
path_wm = os.path.join(path0, 'watermask')
path_indices = os.path.join(path0, 'indices')
path_explanatory_ = os.path.join(path0, 'explanatory')
path_explanatory = {'soil': os.path.join(path_explanatory_, 'soil'),
                    'resampled': os.path.join(path_explanatory_, 'resampled'),
                    'glacier': os.path.join(path_explanatory_, 'glacier'),
                    'wind': os.path.join(path_explanatory_, 'wind')}