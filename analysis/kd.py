'''
Created on Oct 20, 2020

@author: simon
'''
import os
import numpy as np

from asymter import path_indices, read_gdal
from paths import fnexplandict, path_explanatory

maxse = 0.06
gridsize = 513  # 513
gamma = 0.1
logsdict = {
    'ruggedness': True, 'asym': False, 'temp': False, 'prec': True, 'soil': True,
    'absslope': True}
modrrestrict = (200, 700)

def read_longitude(fnexplandict):
    from asymter import geospatial_from_file
    gs = geospatial_from_file(fnexplandict['ruggedness'])
    x = gs.geotrans[3] + np.arange(gs.shape[0]) * gs.geotrans[5]
    y = gs.geotrans[0] + np.arange(gs.shape[1]) * gs.geotrans[1]
    lon = np.arctan2(x[:, np.newaxis], y[np.newaxis, :]) * 180 / np.pi + 45
    lon[lon > 180] -= 360
    return lon

def read_region(fnexplandict):
    lon = read_longitude(fnexplandict)
    reg = np.zeros_like(lon, dtype=np.uint8) + 255
    reg[np.logical_and(lon >= -165, lon < -124)] = 1  # W American
    reg[np.logical_and(lon >= -124, lon < -62)] = 2  # C/E American
    reg[np.logical_and(lon >= -62, lon < -22)] = 3  # Greenland
    reg[np.logical_and(lon >= 3, lon < 60)] = 4  # European
    reg[np.logical_and(lon >= 60, lon < 119)] = 5  # W/C Siberian A
    reg[np.logical_or(lon >= 119, lon < -165)] = 6  # Far Asian E
    return reg

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
            if isinstance(rname, str):
                if rname == 'lon':
                    rdata = read_longitude(fnexplandict)
                elif rname == 'region':
                    rdata = read_region(fnexplandict)
                else:
                    rdata, _, _ = read_gdal(fnexplandict[rname])
            else:
                rdata, _, _ = read_gdal(fnexplandict[rname[0]])
                rdata = rdata[rname[1], ...]
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

def conditional_iqr(pdf, grid, cutoff=0.01, gamma=gamma):
    q1 = conditional_quantile(pdf, grid, quantile=gamma, cutoff=cutoff)
    q3 = conditional_quantile(pdf, grid, quantile=1 - gamma, cutoff=cutoff)
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

def plot_median_relief_temp(pdf, grid, cutoff=0.05):
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
        if 'vlines' in plotdict:
            for v_ in plotdict['vlines']:
                axs[0].axvline(v_, c='#666666', alpha=0.06, lw=0.30)

    if 'xlabel' in plotdict:
        axs[1].text(
            0.5, -0.3, plotdict['xlabel'], transform=axs[1].transAxes,
            ha='center', va='baseline')
    if label is not None:
        axs[0].text(
            0.5, 1.04, label, ha='center', va='baseline', transform=axs[0].transAxes)
    return mps

def plot_kd_soil(fnout, scenname='bandpass'):
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, path_figures
    import colorcet as cc
    from string import ascii_lowercase
    pd = {
        'bgcol': ('#d0d0d0', '#d0d0d0'), 'cmap': (cc.cm['bwy'], cc.cm['CET_CBL1']),
        'vmax': (0.07, 0.3), 'vmin': (-0.07, 0.00), 'ylim': (-17, 3),
        'yticks': (-15, -10, -5, 0), 'cticks': ([-0.05, 0.0, 0.05], [0.0, 0.1, 0.2, 0.3])}
    index = 'logratio'
    cutoff = 0.02
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    fig, axs = prepare_figure(
        nrows=2, ncols=2, figsize=(0.79, 0.82), sharex='col', sharey=True,
        left=0.175, right=0.785, bottom=0.125, top=0.945, wspace=0.2, hspace=0.2,
        remove_spines=False)
    xticks = (30, 100, 1000)
    xmticks = (40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900)
    pd_ = {'xlim': (1.2, 3.0), 'xticks': np.log10(xticks), 'xlabel': 'relief $r$ [m]',
           'xticklabels': xticks, 'xminorticks': np.log10(xmticks)}
    _plot_kd_column(
        axs[:, 0], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'ruggedness'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, restrict=[('soil', 0.00, 0.75)],
        label='thin soil')
    mps = _plot_kd_column(
        axs[:, 1], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'ruggedness'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, restrict=[('soil', 1.5, 1000.0)],
        label='thick soil')
    for ax in axs[:, 0]:
        ax.text(
            -0.63, 0.5, 'temperature $T$ [$^{\\circ}\\mathrm{C}$]',
            transform=ax.transAxes, ha='left', va='center', rotation=90)
    cbpos = {'x': 0.16, 'dx': 0.04, 'dy': 0.24, 'labx':-3.2, 'laby':-4.0, 'y': 0.02}
    clabels = ('\\textbf{median} [-]', '\\textbf{spread} [-]')
    for jax, ax in enumerate(axs[:, -1]):
        pos = ax.properties()['position']
        _y = (0.5 * (pos.y0 + pos.y1) - 0.5 * cbpos['dy'] - cbpos['y'])
        rect_cax = (pos.x1 + cbpos['x'], _y, cbpos['dx'], cbpos['dy'])
        cax = fig.add_axes(rect_cax)
        cbar = fig.colorbar(
            mps[jax], cax=cax, orientation='vertical', ticklocation='left',
            ticks=pd['cticks'][jax])
        ax.text(
            1.40, 0.90, clabels[jax], ha='center', va='baseline', transform=ax.transAxes,
            color='#000000')
    bbox_ = {'facecolor': '#333333', 'edgecolor': 'none', 'boxstyle':'square,pad=0.12',
             'alpha': 0.9}
    for jax, ax in enumerate(axs.flatten(order='F')):
        col = '#ffffff' if jax in (3,) else '#666666'
        bbox = None
        ax.text(
            0.04, 0.90, ascii_lowercase[jax] + ')', ha='left', va='baseline', color=col,
            transform=ax.transAxes, bbox=bbox)
    fig.savefig(os.path.join(path_figures, fnout))

def plot_kd_regions(fnout):
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, path_figures
    import colorcet as cc
    from string import ascii_lowercase
    pd = {
        'bgcol': ('#d0d0d0', '#d0d0d0'), 'cmap': (cc.cm['bwy'], cc.cm['CET_CBL1']),
        'vmax': (0.07, 0.30), 'vmin': (-0.07, 0.00), 'ylim': (-17, 0),
        'yticks': (-15, -10, -5, 0), 'cticks': ([-0.05, 0.0, 0.05], [0.0, 0.1, 0.2, 0.3])}
    scenname = 'bandpass'
    index = 'logratio'
    cutoff = 0.02  # 0.02
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    fig, axs = prepare_figure(
        nrows=2, ncols=4, figsize=(1.50, 0.92), sharex='col', sharey=True,
        left=0.095, right=0.870, bottom=0.110, top=0.945, wspace=0.3, hspace=0.2,
        remove_spines=False)
    xticks = (30, 100, 1000)
    xmticks = (40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900)
    pd_ = {'xlim': (1.4, 3.0), 'xticks': np.log10(xticks), 'xlabel': 'relief $r$ [m]',
           'xticklabels': xticks, 'xminorticks': np.log10(xmticks)}
    _plot_kd_column(
        axs[:, 0], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', 'ruggedness'), gridsize=gridsize, cutoff=cutoff,
        restrict=[('region', 1, 3)], label='America')
    _plot_kd_column(
        axs[:, 1], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', 'ruggedness'), gridsize=gridsize, cutoff=cutoff,
        restrict=[('region', 4, 6)], label='Asia')
    _plot_kd_column(
        axs[:, 2], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', 'ruggedness'), gridsize=gridsize, cutoff=cutoff,
        restrict=[('glacierras', 1, 1)], label='glaciated')
    mps = _plot_kd_column(
        axs[:, 3], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', 'ruggedness'), gridsize=gridsize, cutoff=cutoff,
        restrict=[('glacierras', 0, 0)], label='unglaciated')
    for ax in axs[:, 0]:
        ax.text(
            -0.56, 0.5, 'temperature $T$ [$^{\\circ}\\mathrm{C}$]',
            transform=ax.transAxes, ha='left', va='center', rotation=90)
    cbpos = {'x': 0.10, 'dx': 0.02, 'dy': 0.24, 'labx': 0.5, 'laby':-4.0, 'y': 0.02}
    clabels = ('\\textbf{median} [-]', '\\textbf{spread} [-]')
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
        col = '#ffffff' if jax in (1, 3, 5, 7) else '#666666'
        bbox = bbox_ if jax < 0 else None
        ax.text(
            0.04, 0.90, ascii_lowercase[jax] + ')', ha='left', va='baseline', color=col,
            transform=ax.transAxes, bbox=bbox)
    fig.savefig(os.path.join(path_figures, fnout))
#     plt.show()

def plot_kd_small(fnout, scenname='bandpass', explan='ruggedness'):
    import matplotlib.pyplot as plt
    from plotting import prepare_figure, path_figures
    import colorcet as cc
    from string import ascii_lowercase
    pd = {
        'bgcol': ('#d0d0d0', '#d0d0d0'), 'cmap': (cc.cm['bwy'], cc.cm['CET_CBL1']),
        'vmax': (0.07, 0.30), 'vmin': (-0.07, 0.00), 'ylim': (3, -17),
        'yticks': (-15, -10, -5, 0), 'cticks': ([-0.05, 0.0, 0.05], [0.0, 0.1, 0.2, 0.3])}
    index = 'logratio'
    cutoff = 0.02
    xlabelpos = -0.30
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    fig, axs = prepare_figure(
        nrows=2, ncols=7, figsize=(1.98, 0.82), sharex='col', sharey=True,
        left=0.070, right=0.913, bottom=0.125, top=0.945, wspace=0.2, hspace=0.2,
        remove_spines=False)
    if explan == 'ruggedness':
        xticks, xlim = (100, 1000), (np.log10(40), np.log10(1000))  # 1.2, 3.0
        xmticks = (40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
        xlabel, xlabel_short = 'relief $r$ [m]', '$r$ [m]'
    elif explan == 'absslope':
        xticks, xlim = (0.1, 0.3), (np.log10(0.05), np.log10(0.32))
        xmticks = (0.2, 0.4)
        xlabel, xlabel_short = 'mean slope $s_m$ [-]', '$s_m$ [-]'
    pd_ = {'xlim': xlim, 'xticks': np.log10(xticks), 'xlabel': xlabel,
           'xticklabels': xticks, 'xminorticks': np.log10(xmticks), 'xlabelpos': xlabelpos,
           'vlines': np.log10(np.array(modrrestrict))}
    _plot_kd_column(
        axs[:, 0], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', explan),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff, label='everywhere')

    pd_['xlabel'] = xlabel_short
    _plot_kd_column(
        axs[:, 1], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', explan), gridsize=gridsize, cutoff=cutoff,
        restrict=[('region', 4, 6)], label='Asia')
    _plot_kd_column(
        axs[:, 2], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', explan), gridsize=gridsize, cutoff=cutoff,
        restrict=[('region', 1, 3)], label='America')
    _plot_kd_column(
        axs[:, 5], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', explan), gridsize=gridsize, cutoff=cutoff,
        restrict=[('glacierras', 1, 1)], label='glaciated')
    _plot_kd_column(
        axs[:, 3], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', explan), gridsize=gridsize, cutoff=cutoff,
        restrict=[(('wind', 1), None, 0)], label='northerly wind')
    _plot_kd_column(
        axs[:, 4], fnindex, fnexplandict, {**pd, **pd_}, selimit=selimit,
        explannames=('temp', explan), gridsize=gridsize, cutoff=cutoff,
        restrict=[(('wind', 1), 0, None)], label='southerly wind')
    xticks = (300, 1000)
    xmticks = (100, 200, 400, 500, 600, 700, 800, 900, 1100)
    pd_ = {'xlim': (2.0, 3.2), 'xticks': np.log10(xticks), 'xticklabels': xticks,
           'xminorticks': np.log10(xmticks), 'xlabel': 'precip $P$ [mm]',
           'xlabelpos': xlabelpos}
    mps = _plot_kd_column(
        axs[:, -1], fnindex, fnexplandict, {**pd, **pd_}, explannames=('temp', 'prec'),
        selimit=selimit, gridsize=gridsize, cutoff=cutoff,
        restrict=[('ruggedness', *modrrestrict)], label='moderate relief')

    for ax in axs[:, 0]:
        ax.text(
            -0.67, 0.50, 'temperature $T$ [$^{\\circ}\\mathrm{C}$]',
            transform=ax.transAxes, ha='left', va='center', rotation=90)
    cbpos = {'x': 0.066, 'dx': 0.016, 'dy': 0.240, 'labx': 0.5, 'laby':-4.0, 'y': 0.020}
    clabels = ('\\textbf{median} [-]', '\\textbf{spread} [-]')
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
        col = '#ffffff' if jax in (1, 5, 7, 9 , 11, 13) else '#666666'
        bbox = bbox_ if jax < 0 else None
        ax.text(
            0.04, 0.90, ascii_lowercase[jax] + ')', ha='left', va='baseline', color=col,
            transform=ax.transAxes, bbox=bbox)
    fig.savefig(os.path.join(path_figures, fnout))

def plot_kd_temperature(fnout, scenname='bandpass'):
    import matplotlib.pyplot as plt
    import colorcet as cc
    from plotting import prepare_figure, path_figures
    index = 'logratio'
    cutoff = 0.05
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    restrict = [('ruggedness', *modrrestrict)]
    selimit = (fnindexse, maxse)
    pdf, grid = joint_pdf(
        fnindex, fnexplandict, explannames=('temp',), selimit=selimit, restrict=restrict,
        gridsize=gridsize)
    print(pdf.shape)
    fig, ax = prepare_figure(
        nrows=1, ncols=1, figsize=(6.268, 1.100), figsizeunit='in',
        left=0.0684, right=0.9980, bottom=0.2800, top=0.9900, wspace=0.2, hspace=0.2,
        remove_spines=False)  # left=0.064, figsize=6.71
    ax.set_facecolor('#d0d0d0')
    Z = np.sum(pdf, axis=0)
    pdfc = pdf / Z[np.newaxis, :]
    pdfc[:, Z < cutoff * np.max(Z)] = np.nan
    mp = ax.pcolormesh(
        grid[1], grid[0], pdfc, vmin=0, vmax=np.nanpercentile(pdfc, 99),
        cmap=cc.cm['CET_CBL1'], shading='auto', antialiased=True, lw=-1,
        edgecolor='face')
    mp.set_rasterized(True)
#     med = conditional_quantile(pdf, grid, quantile=0.5)
#     ax.plot(grid[1], med, c='#cccccc', lw=0.5, alpha=0.5)
    ax.axhline(0, c='#666666', lw=0.5, alpha=0.5)
    ax.set_ylim((-0.15, 0.15))
    ax.set_xlim((-17, 3))
    ax.set_xticks([-15, -10, -5, 0])
    ax.text(-0.073, 0.500, '$a$ [-]', rotation=90, va='center', transform=ax.transAxes)
    ax.text(
        0.48, -0.36, 'increasing temperature T [$^{\\circ}\\mathrm{C}$]', ha='center',
        va='baseline', transform=ax.transAxes)
    fig.savefig(os.path.join(path_figures, fnout))

def interrogate_results():
    scenname, index = 'bandpass', 'logratio'
    fnindex = os.path.join(path_indices, scenname, f'{scenname}_{index}.tif')
    fnindexse = os.path.join(path_indices, scenname, f'{scenname}_{index}_se.tif')
    selimit = (fnindexse, maxse)
    im, _, _ = read_gdal(fnindex)
    mask = read_mask(fnexplandict=fnexplandict, selimit=selimit)
    valid = np.logical_and(np.isfinite(im), mask)
    a = im[valid].flatten()
#     print(np.count_nonzero(np.abs(a) < 0.04) / len(a))
#     print(np.count_nonzero(a > 0.04) / np.count_nonzero(np.abs(a) > 0.04))
#     print(np.count_nonzero(a > 0.1) / np.count_nonzero(np.abs(a) > 0.1))
#     print(np.count_nonzero(np.abs(a) > 0.2))
#     print(np.nanpercentile(a, [5, 25, 50, 75, 95]))
    for scenname2 in ['bandpass002', 'bandpass008', 'lowpass']:
        im2, _, _ = read_gdal(os.path.join(
            path_indices, scenname2, f'{scenname2}_{index}.tif'))
        valid_all = np.logical_and(valid, np.isfinite(im2))
        b, b2 = im[valid_all].flatten(), im2[valid_all].flatten()
        print(scenname2)
        print('R2: ', np.corrcoef(b, b2)[0, 1])
        print('RMSE', np.mean((b - b2) ** 2) ** (1 / 2))
        from scipy.stats import spearmanr
        print('Spearman', spearmanr(b, b2)[0])

if __name__ == '__main__':
#     plot_kd(fnout='kde.pdf')
#     plot_kd_regions(fnout='kderegions.pdf')
#     plot_kd_soil(fnout='kdesoil.pdf')
    plot_kd_small(fnout='kdesmall.pdf')
#     plot_kd_small(fnout='kdesmall_slope.pdf', explan='absslope')
#     for scenname in ['lowpass', 'bandpass002', 'bandpass008']:
#         plot_kd_soil(fnout=f'kdesoil_{scenname}.pdf', scenname=scenname)
#         plot_kd_small(fnout=f'kde_small_{scenname}.pdf', scenname=scenname)
#     plot_kd_temperature('kdetemp.pdf')
#     interrogate_results()

