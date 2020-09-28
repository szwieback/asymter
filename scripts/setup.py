'''
Created on Sep 28, 2020

@author: simon
'''

def setup_path():
    import os, sys
    parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, parentdir)

if __name__ == '__main__':
    setup_path()
    from asymter import download_arctic_watermask, download_all_adem_tiles
    download_arctic_watermask()
    download_all_adem_tiles()