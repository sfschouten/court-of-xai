'''Utility functions to generate plots'''

import matplotlib as mpl
mpl.use('PS')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import tight_layout
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple

mpl.style.use('seaborn-poster')
sns.set_palette(sns.color_palette(['#43406b', '#d15a00', '#27f77d']))
# sns.palplot(sns.color_palette(['#43406b', '#d15a00', '#27f77d']))
# sns.set_palette('cubehelix')

font = {'size'   : 17}
mpl.rc('font', **font)

histcolor = '#143f7a'

conscmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#142c89", "#142c89"])

def init_gridspec(nrow, ncol, nax) :
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    axes = []
    for i in range(nax) :
        axes.append(plt.subplot(gs[i//ncol, i%ncol]))

    return fig, axes

def adjust_gridspec() :
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

def show_gridspec() :
    plt.show()

def set_square_aspect(axes) :
    x0,x1 = axes.get_xlim()
    y0,y1 = axes.get_ylim()
    axes.set_aspect(abs(x1-x0)/abs(y1-y0))

def save_axis_in_file(fig, ax, dirname, filename):

    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.png'), bbox_inches=extent, dpi=1000)

    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.svg'), bbox_inches=extent)
        
    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.pdf'), bbox_inches=extent)

def save_table_in_file(table, dirname, filename) :
    table.to_csv(os.path.join(dirname, filename + '.csv'), index=True)

def annotate(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, legend_location='upper left') :
    if xlabel is not None : ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None : ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(labelsize=20)
    if title is not None : ax.set_title(title)
    if xlim is not None : ax.set_xlim(*xlim)
    if ylim is not None : ax.set_ylim(*ylim)

    set_square_aspect(ax)
    sns.despine(ax=ax)
    ax.legend(loc=legend_location)

def generate_correlation_density_plot(ax, correlations: List[Tuple[str, np.ndarray]]):
  linestyles = ['-', '--', ':']
  for (name, measure) in correlations:
    sns.kdeplot(measure, linewidth=2, linestyle=linestyles.pop(), color='k', ax=ax, label=name)
