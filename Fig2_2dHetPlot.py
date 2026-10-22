import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic_2d

# =============================================================================
# 1. PARAMETERS, CONFIGURATIONS, & INDEPENDENT SIZES
# =============================================================================
SWEEP_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure 2/Fig2_HetSweepResults.csv'
OUTPUT_FOLD = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 2/'
HET_COLORMAP = 'plasma'

os.makedirs(OUTPUT_FOLD, exist_ok=True)

# Millimeter conversion factor
mm = 1 / 25.4  

# --- MATCH GEOMETRY STYLES FROM FIG 4 ---
# Total canvas size: 70mm width x 70mm height (square)
FIGURE_SIZE = (60.0 * mm, 70 * mm)

# Core axes positioning leaves room on the right [left, bottom, width, height]
# Adjusted to give 15% margin on right for the vertical colorbar
MAIN_AXES_POS = [0.2, 0.15, 0.75, 0.675]
COLORBAR_AXES_POS = [0.2, 0.85, 0.75, 0.035]
COLORBAR_LABEL_PADDING = 4

# --- MATPLOTLIB RC PARAMS MATCHING FIG 4 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10       
plt.rcParams['xtick.labelsize'] = 7       
plt.rcParams['ytick.labelsize'] = 7       
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# =============================================================================
# 1. LOAD DATA & RECONSTRUCT MATRIX
# =============================================================================
print("Loading sweep results...")
df = pd.read_csv(SWEEP_CSV)

# Extract unique sorted values from your log-spaced sweep parameters
unique_rt = np.sort(df['rt'].unique())
unique_rb = np.sort(df['rb'].unique())

# Pivot the flat sweep rows into a perfect 20x20 grid
# Columns = Transition rate (X), Rows = Branching rate (Y)
pivot_df = df.pivot(index='rb', columns='rt', values='het')
# Sort explicitly to guarantee matrix alignment matches coordinate axes
pivot_df = pivot_df.reindex(index=unique_rb, columns=unique_rt)
heterogeneity_grid = pivot_df.to_numpy()

# =============================================================================
# 2. GENERATE LOG-SPACED GRID EDGES FOR PCOLORMESH
# =============================================================================
# To ensure every single simulation coordinate sits perfectly in the center of 
# its 1-by-1 square block, we must find the outer boundaries (edges) between choices.
def calculate_log_edges(centers):
    log_centers = np.log10(centers)
    log_diffs = np.diff(log_centers)
    # Extrapolate half-steps for the outer boundaries
    log_edges = np.zeros(len(centers) + 1)
    log_edges[1:-1] = log_centers[:-1] + log_diffs / 2
    log_edges[0] = log_centers[0] - log_diffs[0] / 2
    log_edges[-1] = log_centers[-1] + log_diffs[-1] / 2
    return 10**log_edges

rt_edges = calculate_log_edges(unique_rt)
rb_edges = calculate_log_edges(unique_rb)

# =============================================================================
# 3. PLOT HEATMAP
# =============================================================================
print("Generating Figure 2 parameter space heatmap...")
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
ax.set_position(MAIN_AXES_POS)
ax.set_facecolor('white')

# Linear normalization for Heterogeneity H
vmin = df['het'].min()
vmax = df['het'].max()

im = ax.pcolormesh(
    rt_edges, rb_edges, heterogeneity_grid,
    cmap=HET_COLORMAP,
    norm=Normalize(vmin=vmin, vmax=vmax),
    shading='flat',
    edgecolor='none',
    linewidth=0,
    antialiased=False, # Keeps the 20x20 pixel edges sharp and discrete
    rasterized=True
)

# Apply scales and structural labels
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(rt_edges[0], rt_edges[-1])
ax.set_ylim(rb_edges[0], rb_edges[-1])

ax.set_xlabel('Transition rate $r_t$')
ax.set_ylabel('Branching rate $r_b$')

# Styling borders and tick direction matching your Figure 4 profile
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)
#ax.xaxis.set_minor_locator(NullLocator())
#ax.yaxis.set_minor_locator(NullLocator())

# =============================================================================
# 4. HORIZONTAL COLORBAR ON TOP
# =============================================================================
cax = fig.add_axes(COLORBAR_AXES_POS)
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')
cax.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax.xaxis.set_minor_locator(NullLocator())

cbar.set_label('Heterogeneity $H$')

# Save to disk
output_filepath = os.path.join(OUTPUT_FOLD, 'heatmap_parameter_sweep.svg')
plt.savefig(output_filepath, format='svg', dpi=300)
plt.close(fig)

print(f"Successfully generated and saved Figure 2 plot to: {output_filepath}")