'''
Created on Dec 21, 2020

@author: simon
'''
import os
import numpy as np

from asymter import path0
from analysis.plotting import prepare_figure, path_figures

def _load_profile(pname):
    fn = os.path.join(path0, 'profiles', f'{pname}.csv')
    x, z = [], []
    with open(fn, 'r') as f:
        for l in f.readlines():
            try:
                x_, z_ = tuple(l.strip().split('\t'))
                x.append(float(x_))
                z.append(float(z_))
            except:
                pass
    return np.array(x), np.array(z)

def plot_panel(pnames, name=None, deltaz=0.0, yticks=None):
    col = ['#bfaf81', '#092c55']#['#9b8545', '#2d6eb3']
    fig, ax = prepare_figure(
        nrows=1, ncols=1, figsize=(1.520, 0.787), figsizeunit='in', sharex='col',
        sharey=True, left=0.170, right=0.999, bottom=0.225, top=0.99, remove_spines=True)
    for jpname, pname in enumerate(pnames):
        x, z = _load_profile(pname)
        dz = np.min(z) - jpname * deltaz
        ax.plot(x, z - dz, alpha=0.8, c=col[jpname])
    xticks = [0, 2500, 5000]
#     xticklabels = ('0', '2500', '5000')
    ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlim((-500, 7400))
    ypos = -0.26
    ax.text(0.80, ypos, '[m]', transform=ax.transAxes)
    ax.text(-0.05, ypos, 'S', transform=ax.transAxes)
    ax.text(0.93, ypos, 'N', transform=ax.transAxes)

#     ax.text(-0.32, 0.40, 'relative elev. [m]', va='center', transform=ax.transAxes, 
#             rotation=90)
    if name is None: name = pnames[0]
    fig.savefig(os.path.join(path_figures, f'{name}.pdf'))

if __name__ == '__main__':
#     plot_panel(['Melville3', 'Melville2'], name='Melville', deltaz=140.0)
#     plot_panel(
#         ['Magadan3', 'Magadan2'], name='Magadan', deltaz=200.0, yticks=[0, 200, 400])
#     plot_panel(['Goldstream1', 'Goldstream2'], name='Goldstream', deltaz=200)
    plot_panel(['SWAK1', 'SWAK2'], name='SWAK', deltaz=500)

# 60.4650, -153.8621
