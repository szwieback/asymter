'''
Created on Oct 20, 2020

@author: simon
'''
import os
import numpy as np

from asymter import path_indices, path_explanatory, read_gdal

fnexplandict = {'ruggedness': os.path.join(path_indices, 'raw', 'raw_ruggedness.tif'),
                'soil': os.path.join(path_explanatory['resampled'], 'soil.tif'),
                'prec': os.path.join(path_explanatory['resampled'], 'prec.tif'),
                'temp': os.path.join(path_explanatory['resampled'], 'temp10.tif'),
                'glacier': os.path.join(
                    path_explanatory['resampled'], 'glacier_simp.gpkg')}
logsdict = {'ruggedness': True, 'asym': False, 'temp': False, 'prec': True, 'soil': True}

def read_mask(fnexplandict=fnexplandict, selimit=None, erosion_iterations=None):
    from scipy.ndimage import binary_erosion
    mask = np.isfinite(read_gdal(fnexplandict['soil'])[0])
    if erosion_iterations is not None:
        mask = binary_erosion(mask, iterations=erosion_iterations)
    if selimit is not None:
        se, _, _ = read_gdal(selimit[0])
        mask[se > selimit[1]] = False
    return mask

def assemble_data(
        fnindex, fnexplandict, explannames=('soil',), restrict=None):
    explanarr = []
    for en in explannames:
        im, _, _ = read_gdal(fnexplandict[en])
        explanarr.append(im)
    asym, _, _ = read_gdal(fnindex)
    data = np.concatenate((asym[np.newaxis, :], np.array(explanarr)))
    if restrict is not None:
        for rname, rlow, rhigh in restrict:
            rdata, _, _ = read_gdal(fnexplandict[rname])
            if rlow is not None:
                data[:, rdata < rlow] = np.nan
            if rhigh is not None:
                data[:, rdata > rhigh] = np.nan
    return data

def normalize(
        data, logs=None, discard=None, keep_unnormalized=False, mask=None, restrict=None):
    data_ = data.copy()
    if logs is not None:
        for jlog, log in enumerate(logs):
            if log:
                data_[jlog, ...] = np.log10(data_[jlog, ...])
    valid = np.all(np.isfinite(data_), axis=0)
    if mask is not None:
        valid = np.logical_and(valid, mask)
    data_ = data_[:, valid]
    if discard is not None:
        rng = np.random.RandomState(seed=1)
        N = data_.shape[1]
        data_ = data_[:, rng.permutation(N)[int(N * discard):]]
    means = np.mean(data_, axis=1)
    stds = np.std(data_, axis=1)
    if not keep_unnormalized:
        data_ = (data_ - means[:, np.newaxis]) / stds[:, np.newaxis]
    normdict = {
        'data': data_, 'stds': stds, 'means': means, 'valid': valid, 'logs': logs}
    return normdict

def conditional_quantile(pdf, grid, quantile=0.5, cutoff=0.01):
    Z = np.sum(pdf, axis=0)
    marginal = pdf / Z
    axes = 0  # list(range(1, len(pdf.shape) + 1))
    args = np.argmin(np.abs(np.cumsum(marginal, axis=0) - quantile), axis=axes)
    cond_quants = np.vectorize(lambda x: grid[0][x])(args)
    if cutoff is not None:
        cond_quants[Z < cutoff * np.max(Z)] = np.nan
    return cond_quants

def conditional_iqr(pdf, grid, cutoff=0.01):
    q1 = conditional_quantile(pdf, grid, quantile=0.25, cutoff=cutoff)
    q3 = conditional_quantile(pdf, grid, quantile=0.75, cutoff=cutoff)
    return q3 - q1

def joint_pdf(
        fnindex, fnexplandict, gridsize=129, explannames=('soil',), restrict=None,
        selimit=None):
    from fastkde import fastKDE
    data = assemble_data(
        fnindex, fnexplandict, explannames=explannames, restrict=restrict)
    mask = read_mask(fnexplandict=fnexplandict, selimit=selimit)
    logs = [logsdict[en] for en in ['asym'] + list(explannames)]
    dnd = normalize(data, logs=logs, discard=0.0, keep_unnormalized=False, mask=mask)
    grid_norm = [np.linspace(-5, 5, num=gridsize) for _ in range(data.shape[0])]
#     grid_norm[1] = np.linspace(-5, 5, num=257)
    grid = [g * dnd['stds'][jg] + dnd['means'][jg] for jg, g in enumerate(grid_norm)]
    kde = fastKDE.fastKDE(data=dnd['data'].copy(), axes=grid_norm)
    pdf = np.transpose(kde.pdf)
    return pdf, grid

def plot_median_ruggedness_temp(pdf, grid, cutoff=0.05):
    import matplotlib.pyplot as plt
    import colorcet as cc
    cmap = cc.cm['bwy']
    vmax = 0.07
    medianasym = conditional_quantile(pdf, grid, cutoff=cutoff)
    # iqrasym = conditional_iqr(pdf, grid, cutoff=cutoff)
    fig, ax = plt.subplots()
    ax.set_facecolor('#dddddd')
    ax.pcolormesh(
        grid[2], grid[1], medianasym, vmin=-vmax, vmax=vmax, cmap=cmap, shading='auto')
#     ax.pcolormesh(grid[2], grid[1], iqrasym, vmin=0.0, vmax=0.2, cmap='magma')
    ax.set_ylim((-17, 3))
    ax.set_xlim((1.2, 3))
    ax.set_xticks((1.3, 2, 2.3, 3))
    plt.show()

def mutual_information(pdf, normalize=False):
    pdf_ = pdf / np.sum(pdf)
    marg0, marg1 = np.sum(pdf_, axis=0), np.sum(pdf_, axis=1)
    I01 = np.nansum(pdf_ * np.log(pdf_ / (marg0 * marg1)))
    if normalize:
        H0 = -np.nansum(marg0 * np.log(marg0))
        I01 = I01 / H0
    return I01

def _plot_kd_column(
        axs, fnindex, fnexplandict, plotdict, explannames=('temp', 'ruggedness'),
        selimit=None, restrict=None, gridsize=129, cutoff=0.05, label=None):
    mps = []
    pdf, grid = joint_pdf(
        fnindex, fnexplandict, explannames=explannames, selimit=selimit, restrict=restrict,
        gridsize=gridsize)
    metrics = (conditional_quantile(pdf, grid, cutoff=cutoff),
               conditional_iqr(pdf, grid, cutoff=cutoff))
    for jp, (ax, metric) in enumerate(zip(axs, metrics)):
        ax.set_facecolor(plotdict['bgcol'][jp])
        mp = ax.pcolormesh(
            grid[2], grid[1], metric, vmin=plotdict['vmin'][jp], vmax=plotdict['vmax'][jp],
            cmap=plotdict['cmap'][jp], shading='auto', antialiased=True, lw=-1, 
            edgecolor='face')
        mp.set_rasterized(True)
        mps.append(mp)
        if 'ylim' in plotdict: ax.set_ylim(plotdict['ylim'])
        if 'yticks' in plotdict: ax.set_yticks(plotdict['yticks'])
        if 'yticklabels' in plotdict: ax.set_xticklabels(plotdict['yticklabels'])
        if 'xlim' in plotdict: ax.set_xlim(plotdict['xlim'])
        if 'xticks' in plotdict: ax.set_xticks(plotdict['xticks'])
        if 'xticklabels' in plotdict: ax.set_xticklabels(plotdict['xticklabels'])
        if 'xminorticks' in plotdict: ax.set_xticks(plotdict['xminorticks'], minor=True)
    if 'xlabel' in plotdict:
        axs[1].text(
            0.5, -0.25, plotdict['xlabel'], transform=axs[1].transAxes, ha='center',
            va='baseline')
    if label is not None:
        axs[0].text(
            0.5, 1.04, label, ha='center', va='baseline', transform=axs[0].transAxes)
    return mps

def plot_kd(fnout):
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, path_figures
    import colorcet as cc
    from string import ascii_lowercase
    pd = {
        'bgcol': ('#d0d0d0', '#d0d0d0'), 'cmap': (cc.cm['bwy'], cc.cm['CET_CBL1']),
        'vmax': (0.07, 0.15), 'vmin': (-0.07, 0.00), 'ylim': (-17, 3),
        'yticks': (-15, -10, -5, 0), 'cticks': ([-0.05, 0.0, 0.05], [0.0, 0.05, 0.1, 0.15])}
    scenname = 'bandpass004'#'bandpass002'
    index = 'logratio'
    maxse = 0.02
    gridsize = 513
    cutoff = 0.02#0.02
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    fig, axs = prepare_figure(
        nrows=2, ncols=4, figsize=(1.50, 0.92), sharex='col', sharey=True,
        left=0.095, right=0.870, bottom=0.110, top=0.945, wspace=0.3, hspace=0.2, 
        remove_spines=False)
    xticks = (30, 100, 1000)
    xmticks = (40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900)
    pd_ = {'xlim': (1.2, 3), 'xticks': np.log10(xticks), 'xlabel': 'ruggedness $r$ [m]',
           'xticklabels': xticks, 'xminorticks': np.log10(xmticks)}
    _plot_kd_column(
        axs[:, 0], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'ruggedness'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, label='everywhere')
    _plot_kd_column(
        axs[:, 1], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'ruggedness'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, restrict=[('soil', 0.00, 1.0)],
        label='thin soil')
    _plot_kd_column(
        axs[:, 2], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'ruggedness'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, restrict=[('soil', 1.5, 1000.0)],
        label='thick soil')
    xticks = (300, 1000)
    xmticks = (100, 200, 400, 500, 600, 700, 800, 900, 1100)
    pd_ = {'xlim': (2, 3.2), 'xticks': np.log10(xticks), 'xticklabels': xticks,
           'xminorticks': np.log10(xmticks), 'xlabel': 'precipitation $P$ [mm]'}
    mps = _plot_kd_column(
        axs[:, 3], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'prec'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff,
        restrict=[('ruggedness', 200, 800)],  label='rugged terrain')


    for ax in axs[:, 0]:
        ax.text(
            -0.56, 0.5, 'temperature $T$ [$^{\\circ}\\mathrm{C}$]',
            transform=ax.transAxes, ha='left', va='center', rotation=90)
    cbpos = {'x': 0.10, 'dx': 0.02, 'dy': 0.24, 'labx': 0.5, 'laby':-4.0, 'y': 0.02}
    clabels = ('\\textbf{median} [-]','\\textbf{spread} [-]')
    for jax, ax in enumerate(axs[:, -1]):
        pos = ax.properties()['position']
        _y = (0.5 * (pos.y0 + pos.y1) - 0.5 * cbpos['dy'] - cbpos['y'])
        rect_cax = (pos.x1 + cbpos['x'], _y, cbpos['dx'], cbpos['dy'])
        cax = fig.add_axes(rect_cax)
        cbar = fig.colorbar(
            mps[jax], cax=cax, orientation='vertical', ticklocation='left', 
            ticks=pd['cticks'][jax])
        ax.text(
            1.43, 0.90, clabels[jax], ha='center', va='baseline', transform=ax.transAxes,
            color='#000000')
    bbox_ = {'facecolor': '#333333', 'edgecolor': 'none', 'boxstyle':'square,pad=0.12',
             'alpha': 0.9}
    for jax, ax in enumerate(axs.flatten(order='F')):
        col = '#ffffff' if jax in (1, 3, 5) else '#666666'
        bbox = bbox_ if jax == 3 else None
        ax.text(
            0.04, 0.90, ascii_lowercase[jax] + ')', ha='left', va='baseline', color=col,
            transform=ax.transAxes, bbox=bbox)
    fig.savefig(os.path.join(path_figures, fnout))

if __name__ == '__main__':
    plot_kd(fnout='kde.pdf')

