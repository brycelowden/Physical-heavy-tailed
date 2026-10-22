import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from scipy.stats import binned_statistic_2d

# Setup

DATA_CSV    = '[...]' # Use 'Fig4_NeuronData.csv'
OUTPUT_FOLD = '[...]'

densitycolors = ['#ffffff', "#4DBEDE", "#094753"]
DEN_COLORMAP = LinearSegmentedColormap.from_list("customgradient", densitycolors)
HET_COLORMAP = 'plasma'
LINE_WIDTH = 2.0  
COLORBAR_LABEL_PADDING = 5

os.makedirs(OUTPUT_FOLD, exist_ok=True)
mm = 1 / 25.4  

SIZE_HEATMAP = (60 * mm, 70 * mm)
SIZE_LINE = (60 * mm, 70 * mm)
HEATMAP_AXES_POS = [0.18, 0.15, 0.795, 0.67]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10     
plt.rcParams['axes.titlesize'] = 10       
plt.rcParams['figure.titlesize'] = 10     
plt.rcParams['legend.fontsize'] = 9      
plt.rcParams['legend.title_fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 7       
plt.rcParams['ytick.labelsize'] = 7       
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

PLOT_BINS_X, PLOT_BINS_Y = 40, 40
PLOT_X_MIN, PLOT_X_MAX = 0.1, 10.0    
PLOT_Y_MIN, PLOT_Y_MAX = 10, 5000   

df = pd.read_csv(DATA_CSV)

df['branches_per_length'] = df['branches'] / df['length']

clean_df = df[(df['length'] > 0) & (df['branches'] > 0)].dropna(subset=['length', 'branches'])

x_edges = np.logspace(np.log10(PLOT_X_MIN), np.log10(PLOT_X_MAX), PLOT_BINS_X + 1)
y_edges = np.logspace(np.log10(PLOT_Y_MIN), np.log10(PLOT_Y_MAX), PLOT_BINS_Y + 1)


print("Plot 1: 2D Density Heatmap...")
counts, xedges, yedges = np.histogram2d(
    clean_df['length'], clean_df['branches'],
    bins=[x_edges, y_edges]
)

total_neurons = len(clean_df)
density_matrix = counts / total_neurons if total_neurons > 0 else counts
density_matrix[counts < 1] = np.nan  
masked_density = np.ma.masked_invalid(density_matrix)

valid_min_fraction = 1.0 / total_neurons if total_neurons > 0 else 0.0001
valid_max_fraction = np.nanmax(masked_density) if total_neurons > 0 else 1.0

fig1, ax1 = plt.subplots(figsize=SIZE_HEATMAP)
ax1.set_position(HEATMAP_AXES_POS)
ax1.set_facecolor('white')

im1 = ax1.pcolormesh(
    xedges, yedges, masked_density.T,
    cmap=DEN_COLORMAP,
    norm=LogNorm(vmin=valid_min_fraction, vmax=valid_max_fraction),
    shading='flat', edgecolor='face', linewidth=0.01, antialiased=True, rasterized=True
)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
ax1.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
ax1.set_xlabel('Length (mm)')
ax1.set_ylabel('Branches')
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

cax1 = fig1.add_axes([HEATMAP_AXES_POS[0], HEATMAP_AXES_POS[1] + HEATMAP_AXES_POS[3] + 0.015, HEATMAP_AXES_POS[2], 0.035])
cbar1 = fig1.colorbar(im1, cax=cax1, orientation="horizontal")
cax1.xaxis.set_ticks_position('top')
cax1.xaxis.set_label_position('top')
cax1.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax1.xaxis.set_minor_locator(NullLocator())
cbar1.set_label('Density', labelpad=COLORBAR_LABEL_PADDING)

plt.savefig(os.path.join(OUTPUT_FOLD, 'supp_heatmap_1_density.svg'), format='svg', dpi=300)
plt.close(fig1)

print("Plot 2: Average Heterogeneity Heatmap...")
p2_df = clean_df.dropna(subset=['heterogeneity'])

res = binned_statistic_2d(
    p2_df['length'], p2_df['branches'],
    values=p2_df['heterogeneity'],
    statistic='mean',
    bins=[x_edges, y_edges]
)
stat_mean = res.statistic
masked_stat = np.ma.masked_invalid(stat_mean)

valid_min = np.nanmin(stat_mean) if np.any(~np.isnan(stat_mean)) else 0.0
valid_max = np.nanmax(stat_mean) if np.any(~np.isnan(stat_mean)) else 1.0

fig2, ax2 = plt.subplots(figsize=SIZE_HEATMAP)
ax2.set_position(HEATMAP_AXES_POS)
ax2.set_facecolor('white')

im2 = ax2.pcolormesh(
    res.x_edge, res.y_edge, masked_stat.T,
    cmap=HET_COLORMAP,
    norm=Normalize(vmin=valid_min, vmax=valid_max),
    shading='flat', edgecolor='face', linewidth=0.01, antialiased=True, rasterized=True
)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
ax2.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
ax2.set_xlabel('Length (mm)')
ax2.set_ylabel('Branches')
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

cax2 = fig2.add_axes([HEATMAP_AXES_POS[0], HEATMAP_AXES_POS[1] + HEATMAP_AXES_POS[3] + 0.015, HEATMAP_AXES_POS[2], 0.035])
cbar2 = fig2.colorbar(im2, cax=cax2, orientation="horizontal")
cax2.xaxis.set_ticks_position('top')
cax2.xaxis.set_label_position('top')
cax2.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax2.xaxis.set_minor_locator(NullLocator())
cbar2.set_label('Heterogeneity', labelpad=COLORBAR_LABEL_PADDING)

plt.savefig(os.path.join(OUTPUT_FOLD, 'supp_heatmap_2_heterogeneity.svg'), format='svg', dpi=300)
plt.close(fig2)

print("Plot 3: Heterogeneity vs Branches per Length...")
p3_df = df[(df['branches_per_length'] > 0) & (df['heterogeneity'] >= 0)].dropna(subset=['branches_per_length', 'heterogeneity']).copy()

x_limit_min, x_limit_max = 40.0, 500.0
num_bins = 30
min_neurons_per_bin = 25

line_bins = np.logspace(np.log10(x_limit_min), np.log10(x_limit_max), num_bins + 1)
line_bins = np.unique(line_bins)
actual_num_bins = len(line_bins) - 1
bin_centers = np.sqrt(line_bins[:-1] * line_bins[1:])

p3_df['temp_bin'] = pd.cut(p3_df['branches_per_length'], bins=line_bins, labels=False, include_lowest=True)
stats = p3_df.groupby('temp_bin', observed=True)['heterogeneity'].agg(['mean', 'std', 'count']).reset_index()

full_x = np.full(actual_num_bins, np.nan)
full_mean = np.full(actual_num_bins, np.nan)
full_std = np.full(actual_num_bins, np.nan)

for _, row in stats.iterrows():
    bin_idx = int(row['temp_bin'])
    if row['count'] >= min_neurons_per_bin:
        full_x[bin_idx] = bin_centers[bin_idx]
        full_mean[bin_idx] = row['mean']
        full_std[bin_idx] = row['std'] if not np.isnan(row['std']) else 0.0

lower_bound = full_mean - full_std
upper_bound = full_mean + full_std

fig3, ax3 = plt.subplots(figsize=SIZE_LINE, layout='constrained')

plasma_cmap = mpl.colormaps['plasma']
fixed_color = plasma_cmap(0.5)

for i in range(len(full_x) - 1):
    if np.isnan(full_mean[i]) or np.isnan(full_mean[i+1]):
        continue
    ax3.fill_between(
        full_x[i:i+2], lower_bound[i:i+2], upper_bound[i:i+2],
        color=fixed_color, alpha=0.33, edgecolor='none', zorder=2
    )

ax3.plot(full_x, full_mean, color=fixed_color, linewidth=LINE_WIDTH, solid_capstyle='round', zorder=3)

ax3.set_xscale('log')
ax3.set_xlabel('Branches per length (mm$^{-1}$)')
ax3.set_ylabel('Heterogeneity')
ax3.tick_params(which='both', direction='out', bottom=True, left=True, top=False, right=False)
ax3.set_xlim(x_limit_min, x_limit_max)
ax3.set_ylim(0.3, 0.8)
ax3.grid(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.savefig(os.path.join(OUTPUT_FOLD, 'supp_line_plot_branches_per_length.svg'), format='svg')
plt.close(fig3)
