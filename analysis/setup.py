'''
Created on Oct 19, 2020

@author: simon
'''
def setup_path():
    import os, sys
    parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, parentdir)

setup_path()