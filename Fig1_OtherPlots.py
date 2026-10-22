import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator
from tqdm import tqdm

UND_CON_CSV = '[...]'
OUT_FOLDER  = '[...]'
TOP_N = 9
CHOSEN_N = 9

os.makedirs(os.path.dirname(OUT_FOLDER), exist_ok=True)

print("Loading undirected connection matrix...")
df = pd.read_csv(UND_CON_CSV)

#Make df of all cons viewed from all neurons
df_long = pd.concat([
    df[['neuron_id_1','undirected_syn_count']].rename(columns={'neuron_id_1':'neuron_id'}),
    df[['neuron_id_2','undirected_syn_count']].rename(columns={'neuron_id_2':'neuron_id'})
    ],ignore_index=True)

con_counts = df_long.groupby('neuron_id')['undirected_syn_count'].count()

top_N_ids = (con_counts.nlargest(TOP_N).index.tolist())

mm = 1 / 25.4

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

fig, axs = plt.subplots(3,3,figsize=(90 * mm, 90 * mm), constrained_layout = True, sharex = True, sharey=True)
flat_axs = axs.flatten()

chosen_N_ids = top_N_ids.copy()

print(f'\nPlotting chosen {CHOSEN_N} neuron distributions...')

print(chosen_N_ids)

cmap = LinearSegmentedColormap.from_list("customgradient1", ['#00A0CC','#003fa5ff'])

colors = [cmap(i/(CHOSEN_N-1)) for i in range(CHOSEN_N)]

for i, id in enumerate(chosen_N_ids):
    df_neuron = df_long[df_long['neuron_id'] == id].copy()

    syn_counts = df_neuron['undirected_syn_count'].values

    strengths, probabilities = np.unique(syn_counts,return_counts=True)

    num_cons = len(syn_counts)

    probabilities = (probabilities / num_cons)

    ax = flat_axs[i]

    ax.scatter(strengths, probabilities, color=colors[i], s=49, zorder=40, clip_on=False)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1,10**3)
    ax.set_ylim(5*10**(-5),1)
    ax.set_xticks([1,1e2,1e4])
    ax.set_yticks([1e-4,1e-2,1])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

fig.supxlabel('Synapses per connection', fontsize=10)
fig.supylabel('Probability', fontsize=10)

plt.savefig(os.path.join(OUT_FOLDER, 'top_9_neurons.svg'), format='svg')
plt.close(fig)

print(f'\nChosen {CHOSEN_N} neuron distributions saved.')

# Variance plot

tqdm.pandas(desc="Shuffling & Computing Variances")

densitycolors = ['#ffffff', "#00A0CC", "#094553"]
DEN_COLORMAP = LinearSegmentedColormap.from_list("customgradient2", densitycolors)

def compute_real_and_shuffled_var(group):
    syn_counts = group['undirected_syn_count'].values
    k = len(syn_counts)
    
    if k < 2:
        return pd.Series({'real_var': np.nan, 'shuffled_var': np.nan})
    
    S = syn_counts.sum()
    
    real_var = np.var(syn_counts, ddof=1)
    
    shuffled_counts = np.random.multinomial(S, np.full(k, 1.0 / k))
    shuffled_var = np.var(shuffled_counts, ddof=1)
    
    return pd.Series({'real_var': real_var, 'shuffled_var': shuffled_var})

neuron_variances = (
    df_long.groupby('neuron_id')
    .progress_apply(compute_real_and_shuffled_var)
    .reset_index()
)

valid_mask = (neuron_variances['real_var'] > 0) & (neuron_variances['shuffled_var'] > 0)
plot_stats = neuron_variances[valid_mask].copy()

print("\nPlotting Shuffled vs. Real Variance 2D Density Heatmap...")

var_min, var_max = 5e-1, 5e3

x_edges = np.logspace(np.log10(var_min), np.log10(var_max), 100)
y_edges = np.logspace(np.log10(var_min), np.log10(var_max), 100)

counts, xedges, yedges = np.histogram2d(
    plot_stats['real_var'],       # X-axis: Real variance
    plot_stats['shuffled_var'],   # Y-axis: Shuffled variance
    bins=[x_edges, y_edges]
)

total_neurons = len(plot_stats)
density_matrix = counts / total_neurons

density_matrix[counts < 1] = np.nan
masked_density = np.ma.masked_invalid(density_matrix)

valid_min_density = np.nanmin(masked_density)
valid_max_density = np.nanmax(masked_density)

# Compute % of neurons with lower shuffled variance

percent_below = 100 * len(plot_stats[plot_stats['real_var'] >= plot_stats['shuffled_var']]) / len(plot_stats)

print(f'\nPercent of neurons with decreased shuffled variance: {percent_below:.3f}%')

fig_var, ax_var = plt.subplots(figsize=(90 * mm, 90 * mm),constrained_layout=True)

ax_var.set_facecolor('white')

im = ax_var.pcolormesh(
    xedges, yedges, masked_density.T,
    norm=mplcolors.LogNorm(vmin=valid_min_density, vmax=valid_max_density),
    cmap=DEN_COLORMAP,
    shading='flat',
    edgecolor='face',
    linewidth=0.01,
    antialiased=True,
    rasterized=True
)

# Identity line y = x
ax_var.plot([var_min, var_max], [var_min, var_max], color='black', linestyle='--', linewidth=1.0, zorder=5)

ax_var.set_xscale('log')
ax_var.set_yscale('log')
ax_var.set_xlim(var_min, var_max)
ax_var.set_ylim(var_min, var_max)

ax_var.set_xlabel('Variance in synapses')
ax_var.set_ylabel('Variance in synapses (shuffled)')

ax_var.spines['top'].set_visible(False)
ax_var.spines['right'].set_visible(False)
ax_var.tick_params(which='both', direction='out', bottom=True, left=True, top=False, right=False)

CAX_POS = [0.16, 0.85, 0.77 - 0.16, 0.035]

cax_var = fig_var.add_axes(CAX_POS)
cbar_var = fig_var.colorbar(im, cax=cax_var, orientation="horizontal")

cax_var.xaxis.set_ticks_position('top')
cax_var.xaxis.set_label_position('top')
cax_var.tick_params(which='major', direction='in', bottom=False, left=False, top=True, right=False)
cax_var.xaxis.set_minor_locator(NullLocator())
cbar_var.set_label('Density', labelpad=5)

plt.savefig(os.path.join(OUT_FOLDER, 'shuffled_vs_real_variance_heatmap.svg'), format='svg', dpi=300)
plt.close(fig_var)
