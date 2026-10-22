import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic_2d

# Setup

DATA_CSV     = '[...]' # Neuron data 'Fig4_NeuronData.csv'
SIM_DATA_CSV = '[...]' # Simulated neuron data 'Fig2_HetSweepResults.csv'
OUTPUT_FOLD  = '[...]'

densitycolors = ['#ffffff',"#4DBEDE","#094753"]
DEN_COLORMAP = LinearSegmentedColormap.from_list("customgradient",densitycolors)
HET_COLORMAP = 'plasma'
MULTILINE_CMAP = 'cool'
SIM_HEAT_RES_X = 20
SIM_HEAT_RES_Y = 15
SIM_HEAT_MIN_CON = 25
NUM_LINES = 4               
BIN_MIN = 25  
LINE_WIDTH = 2.0 

os.makedirs(OUTPUT_FOLD, exist_ok=True)

mm = 1 / 25.4  

HEATMAP_WIDTH_MM = 44.0
SIM_HEATMAP_WIDTH_MM = 44.0
VERTICAL_PADDING_MM = 5.0
COLORBAR_LABEL_PADDING = 5

SIZE_HEATMAP1 = (70 * mm, 70 * mm)
SIZE_HEATMAP2 = (70 * mm, 70 * mm)
SIZE_HEATMAP_SIM = (56.5 * mm, 56.5 * mm)
SIZE_SIM_LINE = (54.5 * mm, 56.5 / 2 * mm)
SIZE_SIM_LINE_ENV = (56.5 * mm, 56.5 * mm)

SIZE_LINE1        = (56.66 * mm, 60 * mm)  
SIZE_MULTILINE    = (56.66 * mm, 60 * mm)

HEATMAP_AXES_POS = [0.15, 0.15, 0.825, 0.675]
SIM_HEATMAP_AXES_POS = [0.18, 0.17, 0.79, 0.61]

MULTILINE_AXES_POS = [0.23, 0.1875, 0.7325, 0.6]

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

PLOT12_BINS_X, PLOT12_BINS_Y = 40, 40
PLOT12_X_MIN, PLOT12_X_MAX = 25, 2000       
PLOT12_Y_MIN, PLOT12_Y_MAX = 10, 5000    

df = pd.read_csv(DATA_CSV)

sim_df = pd.read_csv(SIM_DATA_CSV)
sim_df['branches_per_connection'] = sim_df['branches'] / sim_df['connections']

# 2d density plot

print("Plot 1: 2D Density Heatmap...")
p1_df = df[(df['connections'] > 0) & (df['branches'] > 0)].dropna(subset=['connections', 'branches'])

p1_x_edges = np.logspace(np.log10(PLOT12_X_MIN), np.log10(PLOT12_X_MAX), PLOT12_BINS_X + 1)
p1_y_edges = np.logspace(np.log10(PLOT12_Y_MIN), np.log10(PLOT12_Y_MAX), PLOT12_BINS_Y + 1)

counts, xedges, yedges = np.histogram2d(
    p1_df['connections'], p1_df['branches'],
    bins=[p1_x_edges, p1_y_edges]
)

total_neurons = len(p1_df)
density_matrix = counts / total_neurons if total_neurons > 0 else counts
density_matrix[counts < 1] = np.nan  
masked_density = np.ma.masked_invalid(density_matrix)

valid_min_fraction = 1.0 / total_neurons if total_neurons > 0 else 0.0001
valid_max_fraction = np.nanmax(masked_density) if total_neurons > 0 else 1.0

fig1, ax1 = plt.subplots(figsize=SIZE_HEATMAP1)
ax1.set_position(HEATMAP_AXES_POS)
ax1.set_facecolor('white')

im1 = ax1.pcolormesh(
    xedges, yedges, masked_density.T,
    cmap=DEN_COLORMAP,
    norm=LogNorm(vmin=valid_min_fraction, vmax=valid_max_fraction),
    shading='flat',
    edgecolor='face',
    linewidth=0.01,
    antialiased=True,
    rasterized=True
)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(PLOT12_X_MIN, PLOT12_X_MAX)
ax1.set_ylim(PLOT12_Y_MIN, PLOT12_Y_MAX)
ax1.set_xlabel('Connections')
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

plt.savefig(os.path.join(OUTPUT_FOLD, 'heatmap_1_density.svg'), format='svg', dpi=300)
plt.close(fig1)

# 2d het plot

print("Plot 2: Average Heterogeneity Heatmap...")
p2_df = df[(df['connections'] > 0) & (df['branches'] > 0)].dropna(subset=['connections', 'branches', 'heterogeneity'])

res = binned_statistic_2d(
    p2_df['connections'], p2_df['branches'],
    values=p2_df['heterogeneity'],
    statistic='mean',
    bins=[p1_x_edges, p1_y_edges]
)
stat_mean = res.statistic
xedges = res.x_edge
yedges = res.y_edge

valid_min = np.nanmin(stat_mean) if np.any(~np.isnan(stat_mean)) else 0.0
valid_max = np.nanmax(stat_mean) if np.any(~np.isnan(stat_mean)) else 1.0
masked_stat = np.ma.masked_invalid(stat_mean)

fig2, ax2 = plt.subplots(figsize=SIZE_HEATMAP2)
ax2.set_position(HEATMAP_AXES_POS)
ax2.set_facecolor('white')

im2 = ax2.pcolormesh(
    xedges, yedges, masked_stat.T,
    cmap=HET_COLORMAP,
    norm=Normalize(vmin=valid_min, vmax=valid_max),
    shading='flat',
    edgecolor='face',
    linewidth=0.01,
    antialiased=True,
    rasterized=True
)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(PLOT12_X_MIN, PLOT12_X_MAX)
ax2.set_ylim(PLOT12_Y_MIN, PLOT12_Y_MAX)
ax2.set_xlabel('Connections')
ax2.set_ylabel('Branches')
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

cax2 = fig2.add_axes([HEATMAP_AXES_POS[0], HEATMAP_AXES_POS[1] + HEATMAP_AXES_POS[3] + 0.015, HEATMAP_AXES_POS[2], 0.05])
cbar2 = fig2.colorbar(im2, cax=cax2, orientation="horizontal")
cax2.xaxis.set_ticks_position('top')
cax2.xaxis.set_label_position('top')
cax2.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax2.xaxis.set_minor_locator(NullLocator())
cbar2.set_label('Heterogeneity', labelpad=COLORBAR_LABEL_PADDING)

plt.savefig(os.path.join(OUTPUT_FOLD, 'heatmap_2_heterogeneity.svg'), format='svg', dpi=300)
plt.close(fig2)

def generate_minimal_line_plot(
    data, x_col, y_col, filename, x_limit_min, x_limit_max, target_fig_size, 
    sim_data=None, sim_y_col='het', sim_num_bins=15, sim_min_per_bin=3,
    num_bins=30, min_neurons_per_bin=25, xlabel='', ylabel=''
):
    filtered_data = data[(data[x_col] >= x_limit_min) & (data[x_col] <= x_limit_max)].copy()
    x_data = filtered_data[x_col]
    
    if len(x_data) == 0:
        return

    bins = np.logspace(np.log10(x_limit_min), np.log10(x_limit_max), num_bins + 1)
    bins = np.unique(bins)
    actual_num_bins = len(bins) - 1
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    filtered_data['temp_bin'] = pd.cut(filtered_data[x_col], bins=bins, labels=False, include_lowest=True)
    stats = filtered_data.groupby('temp_bin', observed=True)[y_col].agg(['mean', 'std', 'count']).reset_index()
    
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
    
    fig, ax = plt.subplots(figsize=target_fig_size, layout='constrained')

    if sim_data is not None:
        filtered_sim = sim_data[(sim_data[x_col] >= x_limit_min) & (sim_data[x_col] <= x_limit_max)].copy()

            sim_bins = np.logspace(np.log10(x_limit_min), np.log10(x_limit_max), sim_num_bins + 1)
            sim_bins = np.unique(sim_bins)
            sim_actual_bins = len(sim_bins) - 1
            sim_bin_centers = np.sqrt(sim_bins[:-1] * sim_bins[1:])
            
            filtered_sim['temp_bin'] = pd.cut(filtered_sim[x_col], bins=sim_bins, labels=False, include_lowest=True)
            sim_stats = filtered_sim.groupby('temp_bin', observed=True)[sim_y_col].agg(['mean', 'std', 'count']).reset_index()
            
            sim_x = np.full(sim_actual_bins, np.nan)
            sim_mean = np.full(sim_actual_bins, np.nan)
            sim_std = np.full(sim_actual_bins, np.nan)
            
            for _, row in sim_stats.iterrows():
                b_idx = int(row['temp_bin'])
                if row['count'] >= sim_min_per_bin:
                    sim_x[b_idx] = sim_bin_centers[b_idx]
                    sim_mean[b_idx] = row['mean']
                    sim_std[b_idx] = row['std'] if not np.isnan(row['std']) else 0.0
            
            sim_lower = sim_mean - sim_std
            sim_upper = sim_mean + sim_std
            
            valid_sim_mask = ~np.isnan(sim_x) & ~np.isnan(sim_lower) & ~np.isnan(sim_upper)
            
            ax.plot(sim_x[valid_sim_mask], sim_lower[valid_sim_mask], color='black', linestyle='--', linewidth=1.0, alpha=0.8, zorder=10)
            ax.plot(sim_x[valid_sim_mask], sim_upper[valid_sim_mask], color='black', linestyle='--', linewidth=1.0, alpha=0.8, zorder=10)

    v_min = min(full_mean) if np.any(~np.isnan(full_mean)) else 0.3
    v_max = max(full_mean) if np.any(~np.isnan(full_mean)) else 0.8
    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap = mpl.colormaps[HET_COLORMAP]

    points = np.array([full_x, full_mean]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    segment_colors = [cmap(norm(full_mean[i])) for i in range(len(full_x) - 1)]

    lc = LineCollection(segments, colors=segment_colors, linewidths=LINE_WIDTH, joinstyle='round', capstyle='round', zorder=3)
    ax.add_collection(lc)

    for i in range(len(full_x) - 1):
        if np.isnan(full_mean[i]) or np.isnan(full_mean[i+1]):
            continue
            
        slice_x = full_x[i:i+2]
        slice_lower = lower_bound[i:i+2]
        slice_upper = upper_bound[i:i+2]
        slice_color = cmap(norm(full_mean[i]))
        
        ax.fill_between(
            slice_x, slice_lower, slice_upper, 
            color=slice_color, alpha=0.33, edgecolor='none', zorder=2
        )

    if sim_data is not None:
        
        fly_color = cmap(norm(np.nanmean(full_mean)))
        handle_fly = Line2D(
            [0], [0], 
            color=fly_color, 
            linewidth=LINE_WIDTH, 
            solid_capstyle='round' 
        )
        
        handle_sim = Line2D(
            [0], [0], 
            color='black', 
            linewidth=1.0, 
            linestyle='--'
        )

        ax.legend(
            handles=[handle_fly, handle_sim],
            labels=['Fly brain', 'Model'],
            loc='upper left',
            frameon=False,
            handlelength=2.0,   
            handletextpad=0.6
        )

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='both', direction='out', bottom=True, left=True, top=False, right=False)
    ax.set_xlim(x_limit_min, x_limit_max)
    ax.set_ylim(0.3, 0.8)
        
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(OUTPUT_FOLD, filename), format='svg')
    plt.close(fig)

def generate_multiline_quantile_plot(data, x_col, y_col, split_col, filename, x_limit_min, x_limit_max, num_bins=30, min_neurons_per_bin=25, xlabel='', ylabel='', cbar_title='', is_sim=False):
    clean_data = data[(data[x_col] > 0) & (data[split_col] > 0)].dropna(subset=[x_col, y_col, split_col]).copy()
    
    clean_data['quantile_bin'] = pd.qcut(clean_data[split_col], q=NUM_LINES, labels=False, duplicates='drop')
    actual_quantiles = clean_data['quantile_bin'].nunique()
    
    fig, ax = plt.subplots(figsize=SIZE_MULTILINE)
    ax.set_position(MULTILINE_AXES_POS)
    
    bins = np.logspace(np.log10(x_limit_min), np.log10(x_limit_max), num_bins + 1)
    bins = np.unique(bins)
    actual_num_bins = len(bins) - 1
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    cmap = mpl.colormaps[MULTILINE_CMAP]
    
    for q_idx in range(actual_quantiles):
        q_data = clean_data[clean_data['quantile_bin'] == q_idx].copy()
        q_data['temp_bin'] = pd.cut(q_data[x_col], bins=bins, labels=False, include_lowest=True)
        
        stats = q_data.groupby('temp_bin', observed=True)[y_col].agg(['mean', 'count']).reset_index()
        
        full_x = np.full(actual_num_bins, np.nan)
        full_mean = np.full(actual_num_bins, np.nan)
        
        for _, row in stats.iterrows():
            b_idx = int(row['temp_bin'])
            if row['count'] >= min_neurons_per_bin:
                full_x[b_idx] = bin_centers[b_idx]
                full_mean[b_idx] = row['mean']
                
        line_color = cmap(q_idx / max(1, actual_quantiles - 1))
        ax.plot(full_x, full_mean, color=line_color, linewidth=LINE_WIDTH, zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='both', direction='out', bottom=True, left=True, top=False, right=False)
    ax.set_xlim(x_limit_min, x_limit_max)
    
    if is_sim:
        ax.set_ylim(0.2,1.0)
    else:
        ax.set_ylim(0.4, 0.75)
        
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    norm = Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    extra_cax = fig.add_axes([MULTILINE_AXES_POS[0], MULTILINE_AXES_POS[1] + MULTILINE_AXES_POS[3] + 0.015, MULTILINE_AXES_POS[2], 0.035])
    cbar = fig.colorbar(sm, cax=extra_cax, orientation="horizontal")
    extra_cax.xaxis.set_ticks_position('top')
    extra_cax.xaxis.set_label_position('top')
    extra_cax.xaxis.set_ticks([0,33.333,66.667,100])
    extra_cax.xaxis.set_ticklabels(['1st','2nd','3rd','4th'])
    extra_cax.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
    extra_cax.xaxis.set_minor_locator(NullLocator())
    cbar.set_label(cbar_title, labelpad=COLORBAR_LABEL_PADDING)
    
    plt.savefig(os.path.join(OUTPUT_FOLD, filename), format='svg')
    plt.close(fig)

def generate_sim_single_line_plot(data, x_col, y_col, filename, xlabel='', ylabel='', target_fig_size=SIZE_LINE1):

    stats = data.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values(by=x_col)
    
    fig, ax = plt.subplots(figsize=target_fig_size, layout='constrained')
    
    x_vals = stats[x_col].to_numpy()
    mean_vals = stats['mean'].to_numpy()
    std_vals = stats['std'].fillna(0.0).to_numpy()
    
    lower_bound = mean_vals - std_vals
    upper_bound = mean_vals + std_vals
    
    fixed_color = "#4DBEDE"
    
    ax.fill_between(x_vals, lower_bound, upper_bound, color=fixed_color, alpha=0.33, edgecolor='none', zorder=2)
    
    ax.plot(x_vals, mean_vals, color=fixed_color, linewidth=LINE_WIDTH, zorder=3)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(data[x_col].min(), data[x_col].max())
    
    valid_lower = lower_bound[lower_bound > 0]
    y_min = valid_lower.min() if len(valid_lower) > 0 else data[y_col].min() * 0.9
    ax.set_ylim(y_min, data[y_col].max() * 1.1)
    
    ax.tick_params(which='both', direction='out', bottom=True, left=True, top=False, right=False)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(OUTPUT_FOLD, filename), format='svg')
    plt.close(fig)


# Plot 3: het vs branches per connection

print("Plot 3: Heterogeneity vs Branches per Connection (With Model Overlay)...")
p3_df = df[(df['branches_per_connection'] > 0) & (df['heterogeneity'] >= 0)].copy()
p3_df = p3_df.dropna(subset=['branches_per_connection', 'heterogeneity'])

generate_minimal_line_plot(
    data=p3_df,
    x_col='branches_per_connection',
    y_col='heterogeneity',
    filename='line_plot_1_branches_per_connection.svg',
    x_limit_min=0.1,    
    x_limit_max=5.0,    
    target_fig_size=SIZE_LINE1, 
    sim_data=sim_df,
    sim_y_col='het',
    num_bins=30,
    min_neurons_per_bin=BIN_MIN,
    xlabel='Branches per connection',
    ylabel='Heterogeneity'
)

# Plot 4: het vs branches multiline

print("Processing Plot 4: Heterogeneity vs Branches (Multiline Quantile Binned)...")
generate_multiline_quantile_plot(
    data=df,
    x_col='branches',
    y_col='heterogeneity',
    split_col='connections',
    filename='multiline_plot_1_branches_by_connections.svg',
    x_limit_min=50,
    x_limit_max=1000,
    num_bins=30,
    min_neurons_per_bin=25,  
    xlabel='Branches',
    ylabel='Heterogeneity',
    cbar_title='Connections quartile'
)


# Plot 5: het vs cons multiline

print("Processing Plot 5: Heterogeneity vs Connections (Multiline Quantile Binned)...")
generate_multiline_quantile_plot(
    data=df,
    x_col='connections',
    y_col='heterogeneity',
    split_col='branches',
    filename='multiline_plot_2_connections_by_branches.svg',
    x_limit_min=30,
    x_limit_max=500,
    num_bins=30,
    min_neurons_per_bin=25,  
    xlabel='Connections',
    ylabel='Heterogeneity',
    cbar_title='Branches quartile'
)


# Simulated neuron plots

print("Sim Plot 1: Branch count vs rb...")
generate_sim_single_line_plot(
    data=sim_df, x_col='rb', y_col='branches',
    filename='sim_plot_1_branches_vs_rb.svg',
    xlabel='Branching rate $r_b$', ylabel='Branches', target_fig_size=SIZE_SIM_LINE
)

print("Sim Plot 2: Connection count vs rt...")
generate_sim_single_line_plot(
    data=sim_df, x_col='rt', y_col='connections',
    filename='sim_plot_2_connections_vs_rt.svg',
    xlabel='Transition rate $r_t$', ylabel='Connections', target_fig_size=SIZE_SIM_LINE
)

print("Sim Plot 3: 2D Heterogeneity Heatmap...")
sim_x_edges = np.logspace(np.log10(SIM_HEAT_MIN_CON), np.log10(sim_df['connections'].max()), SIM_HEAT_RES_X + 1)
sim_y_edges = np.logspace(np.log10(sim_df['branches'].min()), np.log10(sim_df['branches'].max()), SIM_HEAT_RES_Y + 1)

sim_res = binned_statistic_2d(
    sim_df['connections'], sim_df['branches'], values=sim_df['het'],
    statistic='mean', bins=[sim_x_edges, sim_y_edges]
)

fig_sim3, ax_sim3 = plt.subplots(figsize=SIZE_HEATMAP_SIM)
ax_sim3.set_position(SIM_HEATMAP_AXES_POS)
ax_sim3.set_facecolor('white')

im_sim3 = ax_sim3.pcolormesh(
    sim_res.x_edge, sim_res.y_edge, np.ma.masked_invalid(sim_res.statistic).T,
    cmap=HET_COLORMAP, norm=Normalize(vmin=sim_df['het'].min(), vmax=sim_df['het'].max()),
    shading='flat', edgecolor='none', linewidth=0, antialiased=False, rasterized=True
)
ax_sim3.set_xscale('log')
ax_sim3.set_yscale('log')
ax_sim3.set_xlim(sim_x_edges[0], sim_x_edges[-1])
ax_sim3.set_ylim(sim_y_edges[0], sim_y_edges[-1])
ax_sim3.set_xlabel('Connections')
ax_sim3.set_ylabel('Branches')
ax_sim3.spines['top'].set_visible(True)
ax_sim3.spines['right'].set_visible(True)
ax_sim3.tick_params(which='both', direction='in', bottom=True, left=True, top=False, right=False)

cax_sim3 = fig_sim3.add_axes([SIM_HEATMAP_AXES_POS[0], SIM_HEATMAP_AXES_POS[1] + SIM_HEATMAP_AXES_POS[3] + 0.015, SIM_HEATMAP_AXES_POS[2], 0.05])
cbar_sim3 = fig_sim3.colorbar(im_sim3, cax=cax_sim3, orientation="horizontal")
cax_sim3.xaxis.set_ticks_position('top')
cax_sim3.xaxis.set_label_position('top')
cax_sim3.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cbar_sim3.set_label('Heterogeneity', labelpad=COLORBAR_LABEL_PADDING)
plt.savefig(os.path.join(OUTPUT_FOLD, 'sim_plot_3_heatmap_heterogeneity.svg'), format='svg', dpi=300)
plt.close(fig_sim3)

print("Sim Plot 4: Het vs Connections (Multiline)...")
generate_multiline_quantile_plot(
    data=sim_df, x_col='connections', y_col='het', split_col='branches',
    filename='sim_plot_4_het_vs_connections.svg',
    x_limit_min=sim_df['connections'].min(), x_limit_max=sim_df['connections'].max(),
    num_bins=20, min_neurons_per_bin=1,
    xlabel='Connections', ylabel='Heterogeneity', cbar_title='Branches quartile',
    is_sim=True
)

print("Sim Plot 5: Het vs Branches (Multiline)...")
generate_multiline_quantile_plot(
    data=sim_df, x_col='branches', y_col='het', split_col='connections',
    filename='sim_plot_5_het_vs_branches.svg',
    x_limit_min=sim_df['branches'].min(), x_limit_max=sim_df['branches'].max(),
    num_bins=20, min_neurons_per_bin=1,
    xlabel='Branches', ylabel='Heterogeneity', cbar_title='Connections quartile',
    is_sim=True
)

print("Sim Plot 6: Het vs Branches per Connection...")
generate_minimal_line_plot(
    data=sim_df, x_col='branches_per_connection', y_col='het', filename='sim_plot_6_het_vs_bpc.svg',
    x_limit_min=0.1, x_limit_max=5.0,
    target_fig_size=SIZE_SIM_LINE_ENV, num_bins=15, min_neurons_per_bin=3,
    xlabel='Branches per connection', ylabel='Heterogeneity'
)
