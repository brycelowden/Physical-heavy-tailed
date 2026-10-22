import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic_2d

SWEEP_CSV = '[...]' # Output of 'Fig2_2dHetPlotSimulations.py'
OUTPUT_FOLD = '[...]'
HET_COLORMAP = 'plasma'

os.makedirs(OUTPUT_FOLD, exist_ok=True)

mm = 1 / 25.4  

FIGURE_SIZE = (60 * mm, 70 * mm)

MAIN_AXES_POS = [0.2, 0.15, 0.75, 0.675]
COLORBAR_AXES_POS = [0.2, 0.85, 0.75, 0.035]
COLORBAR_LABEL_PADDING = 4

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10       
plt.rcParams['xtick.labelsize'] = 7       
plt.rcParams['ytick.labelsize'] = 7       
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

print("Loading sweep results...")
df = pd.read_csv(SWEEP_CSV)

unique_rt = np.sort(df['rt'].unique())
unique_rb = np.sort(df['rb'].unique())

pivot_df = df.pivot(index='rb', columns='rt', values='het')

pivot_df = pivot_df.reindex(index=unique_rb, columns=unique_rt)
heterogeneity_grid = pivot_df.to_numpy()

def calculate_log_edges(centers):
    log_centers = np.log10(centers)
    log_diffs = np.diff(log_centers)

    log_edges = np.zeros(len(centers) + 1)
    log_edges[1:-1] = log_centers[:-1] + log_diffs / 2
    log_edges[0] = log_centers[0] - log_diffs[0] / 2
    log_edges[-1] = log_centers[-1] + log_diffs[-1] / 2
    return 10**log_edges

rt_edges = calculate_log_edges(unique_rt)
rb_edges = calculate_log_edges(unique_rb)

print("Generating Figure 2 heatmap...")
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
ax.set_position(MAIN_AXES_POS)
ax.set_facecolor('white')

vmin = df['het'].min()
vmax = df['het'].max()

im = ax.pcolormesh(
    rt_edges, rb_edges, heterogeneity_grid,
    cmap=HET_COLORMAP,
    norm=Normalize(vmin=vmin, vmax=vmax),
    shading='flat',
    edgecolor='none',
    linewidth=0,
    antialiased=False,
    rasterized=True
)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(rt_edges[0], rt_edges[-1])
ax.set_ylim(rb_edges[0], rb_edges[-1])

ax.set_xlabel('Transition rate $r_t$')
ax.set_ylabel('Branching rate $r_b$')

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

cax = fig.add_axes(COLORBAR_AXES_POS)
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')
cax.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax.xaxis.set_minor_locator(NullLocator())

cbar.set_label('Heterogeneity $H$')

output_filepath = os.path.join(OUTPUT_FOLD, 'heatmap_parameter_sweep.svg')
plt.savefig(output_filepath, format='svg', dpi=300)
plt.close(fig)
