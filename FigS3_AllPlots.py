import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator
from scipy.stats import binned_statistic_2d

# =============================================================================
# 1. FILE PATHS & CONFIGURATIONS
# =============================================================================
DATA_CSV     = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure 4/Fig4_NeuronData.csv'
LAMINA_TXT   = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure S3/FigS3_LaminaIntrinsic.txt'
OUTPUT_FOLD  = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure S3/'

NEURON_ID_COL = 'id' 
HET_COLORMAP = 'plasma'
COLORBAR_LABEL_PADDING = 5
os.makedirs(OUTPUT_FOLD, exist_ok=True)

# Conversion factor (mm to inches)
mm = 1 / 25.4  

FIGURE_SIZE = (125 * mm, 60 * mm)

# -----------------------------------------------------------------------------
# EXTRA PADDING & POSITION CONFIGURATION (Relative figure fractions 0.0 to 1.0)
# -----------------------------------------------------------------------------
# Format: [left, bottom, width, height]
# To guarantee identical heights, keep the 4th value ('height') matching perfectly across all three elements.
AXES_POSITIONS = {
    'plot1':    [0.1, 0.15, 0.375, 0.8],  # Position for 'All Neurons'
    'plot2':    [0.5, 0.15, 0.375, 0.8],  # Position for 'Lamina Intrinsic'
    'colorbar': [0.8875, 0.15, 0.02, 0.8]   # Position for vertical Colorbar
}

# Exact layout dimensions matching your design specs
PLOT12_BINS_X, PLOT12_BINS_Y = 40, 40
PLOT12_X_MIN, PLOT12_X_MAX = 25, 2000       
PLOT12_Y_MIN, PLOT12_Y_MAX = 10, 5000    

# Global Matplotlib styling presets
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10     
plt.rcParams['axes.titlesize'] = 10       
plt.rcParams['figure.titlesize'] = 10     
plt.rcParams['legend.fontsize'] = 10      
plt.rcParams['legend.title_fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 7       
plt.rcParams['ytick.labelsize'] = 7       
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# =============================================================================
# 2. DATA LOADING & PRE-PROCESSING
# =============================================================================
print("Loading core neuron dataset...")
df = pd.read_csv(DATA_CSV)

# Isolate valid baseline distribution points
base_df = df[(df['connections'] > 0) & (df['branches'] > 0)].dropna(subset=['connections', 'branches', 'heterogeneity'])

# Define common log-edges shared across both binned spaces
x_edges = np.logspace(np.log10(PLOT12_X_MIN), np.log10(PLOT12_X_MAX), PLOT12_BINS_X + 1)
y_edges = np.logspace(np.log10(PLOT12_Y_MIN), np.log10(PLOT12_Y_MAX), PLOT12_BINS_Y + 1)

# Compute the baseline statistical grid to extract uniform color boundaries
res_base = binned_statistic_2d(
    base_df['connections'], base_df['branches'],
    values=base_df['heterogeneity'], statistic='mean', bins=[x_edges, y_edges]
)
stat_mean_base = res_base.statistic

# Fixed reference color parameters captured globally
global_vmin = np.nanmin(stat_mean_base) if np.any(~np.isnan(stat_mean_base)) else 0.0
global_vmax = np.nanmax(stat_mean_base) if np.any(~np.isnan(stat_mean_base)) else 1.0

# Isolate Lamina Intrinsic subclass neurons
stat_mean_lamina = np.full_like(stat_mean_base, np.nan)
if os.path.exists(LAMINA_TXT):
    print("Loading Lamina Intrinsic subclass neuron IDs...")
    with open(LAMINA_TXT, 'r') as f:
        lamina_ids = [int(x.strip()) for x in f.read().split(',') if x.strip().isdigit()]
    
    lamina_df = base_df[base_df[NEURON_ID_COL].isin(lamina_ids)]
    print(f"Filtered down to {len(lamina_df)} active Lamina Intrinsic neurons.")
    
    if len(lamina_df) > 0:
        res_lamina = binned_statistic_2d(
            lamina_df['connections'], lamina_df['branches'],
            values=lamina_df['heterogeneity'], statistic='mean', bins=[x_edges, y_edges]
        )
        stat_mean_lamina = res_lamina.statistic
else:
    print(f"Warning: Could not locate the ID reference file at '{LAMINA_TXT}'. Plot 2 will be blank.")


# =============================================================================
# 3. SIDE-BY-SIDE GRID COMPILATION
# =============================================================================
print("Compiling supplemental side-by-side layout figure...")
fig = plt.figure(figsize=FIGURE_SIZE)

# Setup uniform color normalizer
norm = Normalize(vmin=global_vmin, vmax=global_vmax)

# --- Subplot 1: All Neurons ---
ax1 = fig.add_axes(AXES_POSITIONS['plot1'])
masked_base = np.ma.masked_invalid(stat_mean_base)
im1 = ax1.pcolormesh(
    x_edges, y_edges, masked_base.T,
    cmap=HET_COLORMAP, norm=norm, shading='flat',
    edgecolor='face', linewidth=0.01, antialiased=True, rasterized=True
)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(PLOT12_X_MIN, PLOT12_X_MAX)
ax1.set_ylim(PLOT12_Y_MIN, PLOT12_Y_MAX)
ax1.set_xlabel('Connections')
ax1.set_ylabel('Branches')  # Only included on the far left plot
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

# --- Subplot 2: Lamina Intrinsic Neurons ---
ax2 = fig.add_axes(AXES_POSITIONS['plot2'])
masked_lamina = np.ma.masked_invalid(stat_mean_lamina)
im2 = ax2.pcolormesh(
    x_edges, y_edges, masked_lamina.T,
    cmap=HET_COLORMAP, norm=norm, shading='flat',
    edgecolor='face', linewidth=0.01, antialiased=True, rasterized=True
)
ax2.set_xscale('log')
ax2.set_yscale('log') # Explicit log scale here handles sharey replacement natively
ax2.set_xlim(PLOT12_X_MIN, PLOT12_X_MAX)
ax2.set_ylim(PLOT12_Y_MIN, PLOT12_Y_MAX)
ax2.set_xlabel('Connections')
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
# Suppress the string labels for the shared Y axis while preserving internal ticks
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax2.yaxis.set_minor_formatter(plt.NullFormatter())
ax2.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

# --- Shared Far-Right Colorbar ---
cax = fig.add_axes(AXES_POSITIONS['colorbar'])
cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
cbar.ax.tick_params(which='major', direction='in', left=False, right=True)
cbar.ax.yaxis.set_minor_locator(NullLocator())
cbar.set_label('Heterogeneity', labelpad=COLORBAR_LABEL_PADDING)

# Save merged visual artifact
plt.savefig(os.path.join(OUTPUT_FOLD, 'supp2_side_by_side_heatmaps.svg'), format='svg', dpi=300)
plt.close(fig)

print(f"\nSuccess! Merged panel vector written to destination folder: {OUTPUT_FOLD}")