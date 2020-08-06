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

pathADEM = os.path.join(path0, 'ArcticDEM')
pathwm = os.path.join(path0, 'watermask')
