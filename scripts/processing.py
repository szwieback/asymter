'''
Created on Sep 28, 2020

@author: simon
'''
from setup import setup_path
setup_path()
from asymter import batch_asymter

if __name__ == '__main__':
    spacing_hr = (5e3, -5e3)
    bp = (100, 2000)
    cellsize = (25e3, 25e3)
    indtypes = [
        'median', 'logratio', 'roughness', 'medianEW', 'logratioEW', 'N', 'N_logratio']
    indtypes_dem = ['ruggedness']
    batch_asymter(
        'bandpass', indtypes=indtypes, bp=bp, cellsize=cellsize, bootstrap_se=True)
    batch_asymter(
        'lowpass', indtypes=indtypes, bp=(bp[0], None), cellsize=cellsize, 
        bootstrap_se=True)
    batch_asymter(
        'raw', indtypes=indtypes_dem, cellsize=cellsize, bp=bp, bootstrap_se=False, 
        noslope=True)
    batch_asymter('bandpass002', indtypes=indtypes, cellsize=cellsize, bp=bp,
        bootstrap_se=True, minslope=0.02)
    batch_asymter('bandpass008', indtypes=indtypes, cellsize=cellsize, bp=bp,
        bootstrap_se=True, minslope=0.08)
#     batch_asymter('bandpass000', indtypes=indtypes, cellsize=(25e3, 25e3), bp=(100, 2000),
#         water_cutoffpct=5.0, overwrite=False, bootstrap_se=True, minslope=0.0,
#         N_bootstrap=25, n_jobs=8)
#     batch_asymter('bandpass_hr', indtypes=indtypes, cellsize=(10e3, 10e3), bp=(100, 2000),
#         spacing=spacing_hr, water_cutoffpct=5.0, overwrite=False, bootstrap_se=True,
#         N_bootstrap=25, n_jobs=8)
