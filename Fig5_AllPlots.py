import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy import odr
from scipy.special import zeta

# =============================================================================
# 1. PARAMETERS, FILE CONFIGURATIONS, & INDEPENDENT SIZES
# =============================================================================
DATA_CSV      = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure 5/Fig5_SortedNeurons_Stage2.csv'
CONN_DATA_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireUndCons.csv'
OUT_FOLD      = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 5/'

NEURON_ID_COL  = 'id'  
CONN_PRE_COL   = 'neuron_id_1'
CONN_POST_COL  = 'neuron_id_2'
CONN_SYN_COL   = 'undirected_syn_count'

VIOLIN_CMAP    = 'plasma'
DOT_CMAP       = 'turbo'
NT_CMAP        = 'cool'   
NUM_QUANT      = 20
DOT_SIZE       = 100
BIG_DOT_SIZE   = 100    
MIN_BIN_NEURONS = 25  

# --- CUSTOM NEUROTRANSMITTER COLORS --- #
cmap_nt = mpl.colormaps[NT_CMAP]
NT_COLOR_DICT = {
    'ACH': "#00d5ff",
    'GABA': '#ff00ff',
    'GLUT': cmap_nt(0.5), 
    'Modulatory': '#aaaaaa' 
}

NT_LABELS = {
    'ACH': 'Excitatory',
    'GABA': 'Inhibitory',
    'GLUT': 'GLUT',
    'Modulatory': 'Modulatory'
}

os.makedirs(OUT_FOLD, exist_ok=True)

mm = 1 / 25.4

block = 180/7 * mm

SIZE_DOT_NP     = (60 * mm, 35 * mm)   
SIZE_DOT_NT     = (60 * mm, 35 * mm)   
SIZE_QUANT      = (60 * mm, 70 * mm)   

# Bounding boxes
SIZE_NP_MOSAIC  = (4 * block, 3 * block)
SIZE_NT_MOSAIC  = (3 * block, 3 * block)
SIZE_NP_MULTILINE = (120 * mm, 60 * mm)
SIZE_NT_MULTILINE = (60 * mm, 60 * mm)

DOT_PLOT_XTICKS = {
    'neuropil':        {'xlim': (None, None), 'ticks': None}, 
    'neurotransmitter':{'xlim': (None, None), 'ticks': None}, 
    'reciprocity':     {'xlim': (None, None), 'ticks': None},
    'olfactory':       {'xlim': (None, None), 'ticks': None},
    'optic':           {'xlim': (None, None), 'ticks': None}
}

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

print("Loading consolidated dataset...")
df = pd.read_csv(DATA_CSV)

if 'neuropil' in df.columns:
    df['neuropil'] = df['neuropil'].str.replace('_', ' ')

if 'neurotransmitter_type' in df.columns:
    df['neurotransmitter_type'] = df['neurotransmitter_type'].replace(['OCT', 'SER', 'DA'], 'Modulatory')

# =============================================================================
# 2. PRE-COMPUTE SORTED NEUROPIL COLORS (SORTED BY BRANCHES PER CONNECTION)
# =============================================================================
ignored_neuropils = ['Other', 'Ocelli']
df_np_clean = df[df['neuropil'].notna() & (~df['neuropil'].isin(ignored_neuropils))].copy()

sorted_np_means = (df_np_clean.groupby('neuropil')['branches_per_connection']
                   .mean()
                   .sort_values()
                   .index.tolist())

cmap_violins = mpl.colormaps[VIOLIN_CMAP]

#Normalize colors between cmap(0.25) and cmap(0.95)
neuropil_colors = {np_name: cmap_violins(0.25 + 0.7*(i/(len(sorted_np_means)-1))) for i, np_name in enumerate(sorted_np_means)}

# =============================================================================
# HELPER PLOTTING INSTANCE: Standard Scatter Framework
# =============================================================================
def create_base_scatter_fig(target_size, config_key,y_tick_spacing):
    fig, ax = plt.subplots(figsize=target_size, layout='constrained')
    ax.set_xscale('linear')  
    ax.set_xlabel('Branches per connection')
    ax.set_ylabel('Heterogeneity')
    
    decimal_formatter = ticker.FuncFormatter(lambda x, _: '{:g}'.format(x))
    ax.xaxis.set_major_formatter(decimal_formatter)
    
    cfg = DOT_PLOT_XTICKS.get(config_key, {'xlim': (None, None), 'ticks': None})
    if cfg['xlim'] and cfg['xlim'][0] is not None:
        ax.set_xlim(cfg['xlim'])
    if cfg['ticks'] is not None:
        ax.xaxis.set_major_locator(ticker.FixedLocator(cfg['ticks']))
    
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    
    # Enforce consistent y-axis major tick mark intervals of 0.02 across all dot plots
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', which='both')
    return fig, ax


# =============================================================================
# 3. PLOT 2: Neuropil 2D Subpopulation Dot Plot
# =============================================================================
print("Generating Plot 2: Neuropil Scatter Coordinate Mapping...")
fig2, ax2 = create_base_scatter_fig(SIZE_DOT_NP, 'neuropil',0.04)

for np_name in sorted_np_means:
    sub_df = df_np_clean[df_np_clean['neuropil'] == np_name]
    if len(sub_df) > 0:
        avg_bpc = sub_df['branches_per_connection'].mean()
        avg_het = sub_df['heterogeneity'].mean()
        ax2.scatter(avg_bpc, avg_het, color=neuropil_colors[np_name], s=BIG_DOT_SIZE, edgecolor='none', zorder=3, clip_on=False)
        ax2.text(avg_bpc, avg_het, str(np_name), fontsize=1, fontname='Arial', zorder=4)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

plt.savefig(os.path.join(OUT_FOLD, 'fig5_2_neuropil_dots.svg'), format='svg')
plt.close(fig2)


# =============================================================================
# 4. PLOT 3: Neurotransmitter 2D Dot Plot sorted by Heterogeneity
# =============================================================================
print("Generating Plot 3: Neurotransmitter Scatter Coordinates...")
fig3, ax3 = create_base_scatter_fig(SIZE_DOT_NT, 'neurotransmitter',0.02)

df_nt_clean = df[df['neurotransmitter_type'].notna()].copy()

#sorted_nt_means = (df_nt_clean.groupby('neurotransmitter_type')['heterogeneity'].mean().sort_values().index.tolist())
sorted_nt_means = ['ACH','GLUT','GABA','Modulatory']

print(sorted_nt_means)

num_nts = max(1, len(sorted_nt_means) - 1)
nt_colors = {}
for i, nt in enumerate(sorted_nt_means):
    color = NT_COLOR_DICT.get(nt)
    if color is None:
        color = cmap_nt(i / num_nts)
    nt_colors[nt] = color

for nt in sorted_nt_means:
    sub_df = df_nt_clean[df_nt_clean['neurotransmitter_type'] == nt]
    if len(sub_df) > 0:
        avg_bpc = sub_df['branches_per_connection'].mean()
        avg_het = sub_df['heterogeneity'].mean()
        
        ax3.scatter(avg_bpc, avg_het, color=nt_colors[nt], s=BIG_DOT_SIZE, edgecolor='none', zorder=3, clip_on=False, label=NT_LABELS.get(nt, nt))
        ax3.text(avg_bpc, avg_het, str(nt), fontsize=1, fontname='Arial', zorder=4)

plt.savefig(os.path.join(OUT_FOLD, 'fig5_3_neurotransmitter_dots.svg'), format='svg')
plt.close(fig3)


# =============================================================================
# HELPER METHOD: Quantile Profiling and Colorbar Engine
# =============================================================================
def generate_quantile_dot_plot(dataframe, split_col, filename, base_title, target_size, config_key, drop_zeros=False, exclude_highest_bin=False, y_tick_spacing=0.02):
    sub_df = dataframe[dataframe[split_col].notna() & 
                       dataframe['branches_per_connection'].notna() & 
                       dataframe['heterogeneity'].notna()].copy()
    
    if drop_zeros:
        sub_df = sub_df[sub_df[split_col] > 0].copy()
        
    if len(sub_df) == 0:
        return

    q = NUM_QUANT
    if exclude_highest_bin:
        q += 1
    
    sub_df['bin'] = pd.qcut(sub_df[split_col], q=q, labels=False, duplicates='drop')
    grouped = sub_df.groupby('bin').agg({'branches_per_connection': 'mean', 'heterogeneity': 'mean'}).reset_index()
    
    if exclude_highest_bin:
        highest_bin_id = grouped['bin'].max()
        grouped = grouped[grouped['bin'] != highest_bin_id].reset_index(drop=True)

    max_bin_idx = int(grouped['bin'].max())
    norm = Normalize(vmin=0, vmax=100)
    cmap_dots = mpl.colormaps[DOT_CMAP]
    
    fig, ax = create_base_scatter_fig(target_size, config_key,y_tick_spacing)
    
    for _, row in grouped.iterrows():
        bin_val = int(row['bin'])
        bin_percentile = (bin_val / max_bin_idx) * 100 if max_bin_idx > 0 else 0
        color = cmap_dots(norm(bin_percentile))
        ax.scatter(row['branches_per_connection'], row['heterogeneity'], color=color, s=DOT_SIZE, edgecolor='none', zorder=3, clip_on=False)
        
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_dots)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, location='top', shrink=1.0, aspect=20, pad=0.04)
    cbar.ax.tick_params(which='both', direction='in', labelsize=6)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    
    cbar_title = f"{base_title} percentile"
    cbar.set_label(cbar_title, labelpad=4.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(OUT_FOLD, filename), format='svg')
    plt.close(fig)


# =============================================================================
# 5. PLOTS 4 THROUGH 14: Continuous Quantile Matrices
# =============================================================================
print("Generating Plot 4: Reciprocity Quantile Groups...")
generate_quantile_dot_plot(df, 'reciprocity', 'fig5_4_reciprocity_dots.svg', 'Reciprocity', SIZE_QUANT, 'reciprocity',y_tick_spacing=0.02)

print("Generating Plot 5: Olfactory Distance Quantile Groups...")
generate_quantile_dot_plot(df, 'distance_olfactory', 'fig5_5_olfactory_distance_dots.svg', 'Olfactory distance', SIZE_QUANT, 'olfactory', drop_zeros=True)

print("Generating Plot 6: Optic Distance Quantile Groups...")
generate_quantile_dot_plot(df, 'distance_optic', 'fig5_6_optic_distance_dots.svg', 'Optic distance', SIZE_QUANT, 'optic', drop_zeros=True)

print("Generating Plot 7: Clustering Coefficient...")
generate_quantile_dot_plot(df, 'clustering_coefficient', 'fig5_7_clustering_coefficient_dots.svg', 'Clustering coefficient', SIZE_QUANT, 'optic',y_tick_spacing=0.01)

print("Generating Plot 8: Closeness Centrality...")
generate_quantile_dot_plot(df, 'closeness_centrality', 'fig5_8_closeness_centrality_dots.svg', 'Closeness centrality', SIZE_QUANT, 'optic')

print("Generating Plot 9: Betweenness Centrality...")
generate_quantile_dot_plot(df, 'betweenness_centrality', 'fig5_9_betweenness_centrality_dots.svg', 'Betweenness centrality', SIZE_QUANT, 'optic')

print("Generating Plot 10: In/Out ratio...")
generate_quantile_dot_plot(df, 'in_out', 'fig5_10_in_out_dots.svg', 'In/out ratio', SIZE_QUANT, 'optic')

print("Generating Plot 11: PageRank...")
generate_quantile_dot_plot(df, 'pagerank', 'fig5_11_pagerank_dots.svg', 'PageRank', SIZE_QUANT, 'optic')

print("Generating Plot 12: K-Shell...")
generate_quantile_dot_plot(df, 'k_shell', 'fig5_12_kshell_dots.svg', 'k-shell index', SIZE_QUANT, 'optic')

print("Generating Plot 13: Hub score...")
generate_quantile_dot_plot(df, 'hub_score', 'fig5_13_hubscore_dots.svg', 'Hub score', SIZE_QUANT, 'optic')

print("Generating Plot 14: Authority score...")
generate_quantile_dot_plot(df, 'authority_score', 'fig5_14_authorityscore_dots.svg', 'Authority score', SIZE_QUANT, 'optic')


# =============================================================================
# HELPER METHOD: ODR Fitting (Figure 3 Style Logic)
# =============================================================================
def fit_connection_odr(syn_counts):
    unique_counts, counts = np.unique(syn_counts, return_counts=True)
    sort_idx = np.argsort(unique_counts)
    xdata = unique_counts[sort_idx].astype(float)
    ydata = counts[sort_idx] / len(syn_counts)
    
    def log_model(p, x):
        alpha, gamma = p[0], p[1]
        if alpha <= 0 or gamma <= 2:
            return np.ones_like(x) * -np.inf
        return np.log10(alpha**gamma) - np.log10(zeta(gamma, 1.0/alpha) - alpha**gamma) - gamma * np.log10(1.0 + alpha * (10**x))
    
    X_log, Y_log = np.log10(xdata), np.log10(ydata)
    odr_model = odr.Model(log_model)
    odr_data = odr.Data(X_log, Y_log)
    fit_odr = odr.ODR(odr_data, odr_model, beta0=[1.0 / xdata.mean(), 2.5])
    fit_output = fit_odr.run()
    return xdata, ydata, fit_output.beta[0], fit_output.beta[1]


# =============================================================================
# HELPER METHOD: Generate Custom Multiline Plots (Figure 4 Style Logic)
# =============================================================================
def generate_category_multiline_plot(dataframe, category_col, active_categories, color_mapping, filename, target_size, xlabel):
    fig, ax = plt.subplots(figsize=target_size, layout='constrained')
    
    for cat in active_categories:
        cat_df = dataframe[dataframe[category_col] == cat].copy()
        if len(cat_df) < MIN_BIN_NEURONS:
            continue
            
        x_min, x_max = cat_df['branches_per_connection'].min(), cat_df['branches_per_connection'].max()
        if x_min == x_max or np.isnan(x_min): continue
        
        bins = np.logspace(np.log10(x_min), np.log10(x_max), 20)
        bins = np.unique(bins)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        
        cat_df['temp_bin'] = pd.cut(cat_df['branches_per_connection'], bins=bins, labels=False, include_lowest=True)
        stats = cat_df.groupby('temp_bin', observed=True)['heterogeneity'].agg(['mean', 'count']).reset_index()
        
        full_x, full_mean = [], []
        for _, row in stats.iterrows():
            b_idx = int(row['temp_bin'])
            if row['count'] >= MIN_BIN_NEURONS:
                full_x.append(bin_centers[b_idx])
                full_mean.append(row['mean'])
                
        if len(full_x) > 1:
            ax.plot(full_x, full_mean, color=color_mapping[cat], linewidth=2.0, label=cat, zorder=3)
            
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Heterogeneity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', which='both')
    
    plt.savefig(os.path.join(OUT_FOLD, filename), format='svg')
    plt.close(fig)


# =============================================================================
# DATA PREPARATION & MOSAIC GENERATION
# =============================================================================
print("\nProcessing New Figure 5 Additions...")
if os.path.exists(CONN_DATA_CSV):
    df_conn = pd.read_csv(CONN_DATA_CSV)
    
    neuron_np_map = df.set_index(NEURON_ID_COL)['neuropil'].to_dict()
    neuron_nt_map = df.set_index(NEURON_ID_COL)['neurotransmitter_type'].to_dict()
    
    df_conn['pre_np'] = df_conn[CONN_PRE_COL].map(neuron_np_map)
    df_conn['post_np'] = df_conn[CONN_POST_COL].map(neuron_np_map)
    df_conn['pre_nt'] = df_conn[CONN_PRE_COL].map(neuron_nt_map)
    df_conn['post_nt'] = df_conn[CONN_POST_COL].map(neuron_nt_map)

    # --- 1. Neuropil Connection Strength Mosaic (6x2) ---
    print("Generating Plot 15: Neuropil Fit Mosaic (4x3)...")
    fig_np_mos, axes_np = plt.subplots(3, 4, figsize=SIZE_NP_MOSAIC, sharex=True, sharey=True, layout='constrained')
    axes_flat_np = axes_np.flatten()
    
    fig_np_mos.supxlabel('Synapses per connection', fontsize=10)
    fig_np_mos.supylabel('Probability', fontsize=10)
    
    for idx, np_name in enumerate(sorted_np_means):
        ax = axes_flat_np[idx]
        sub_conn = df_conn[(df_conn['pre_np'] == np_name) | (df_conn['post_np'] == np_name)]
        syn_counts = sub_conn[CONN_SYN_COL].dropna().to_numpy()
        
        if len(syn_counts) > 10:
            xdata, ydata, alpha_fit, gamma_fit = fit_connection_odr(syn_counts)
            ax.scatter(xdata, ydata, color=neuropil_colors[np_name], s=20, alpha=1.0, zorder=2)
            fitx = np.logspace(0, 4, 200)
            fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit)) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)
            ax.plot(fitx, fity, 'k-', linewidth=2, zorder=3)

            fit_rbrt = (gamma_fit - 1)/(gamma_fit - 2)
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1, 10**4)
        # Keeps all major tick marks every decade
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
        #ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
        # Custom Formatter: Labels only 10^0, 10^2, 10^4. Leaves 10^1 and 10^3 unlabeled.
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: f"$10^{{{int(np.round(np.log10(x)))}}}$" if x > 0 and int(np.round(np.log10(x))) in {0, 2, 4} else ""
        ))
        
        ax.set_ylim(10**(-8), 10**0)
        # Keeps all major tick marks every decade
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
        #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
        # Custom Formatter: Labels only 10^-6, 10^-3, 10^0.
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: f"$10^{{{int(np.round(np.log10(y)))}}}$" if y > 0 and int(np.round(np.log10(y))) in {-8, -4, 0} else ""
        ))

        ax.tick_params(direction='out', which='minor', length=1.5, width=0.25)
        
        # Hardcoded multi-line replacement rule for Periesophageal NP; standard breakdown for others
        if "Periesophageal" in np_name:
            wrapped_name = "Perieso-\nphageal\nNP"
        elif " " in np_name:
            words = np_name.split(" ", 1)
            wrapped_name = f"{words[0]}\n{words[1]}"
        else:
            wrapped_name = np_name
            
        # Box background face and edge alphas explicitly muted to force 100% transparency
        ax.text(0.95, 0.95, wrapped_name, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='none', edgecolor='none'))
        
        """
        # Box for fit rb/rt
        ax.text(0.05, 0.05, f'$r_b/r_t$\n$=${fit_rbrt:.2f}', transform=ax.transAxes, fontsize=5,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='none', edgecolor='none'))
        """
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.savefig(os.path.join(OUT_FOLD, 'fig5_15_neuropil_mosaic.svg'), format='svg')
    plt.close(fig_np_mos)

    # --- 2. Neurotransmitter Connection Strength Mosaic (2x2) ---
    print("Generating Plot 16: Neurotransmitter Fit Mosaic (2x2)...")
    fig_nt_mos, axes_nt = plt.subplots(2, 2, figsize=SIZE_NT_MOSAIC, sharex=True, sharey=True, layout='constrained')
    axes_flat_nt = axes_nt.flatten()
    
    fig_nt_mos.supxlabel('Synapses per connection', fontsize=10)
    fig_nt_mos.supylabel('Probability', fontsize=10)

    for idx, nt_name in enumerate(sorted_nt_means):
        ax = axes_flat_nt[idx]
        sub_conn = df_conn[(df_conn['pre_nt'] == nt_name) | (df_conn['post_nt'] == nt_name)]
        syn_counts = sub_conn[CONN_SYN_COL].dropna().to_numpy()
        
        if len(syn_counts) > 10:
            xdata, ydata, alpha_fit, gamma_fit = fit_connection_odr(syn_counts)
            ax.scatter(xdata, ydata, color=nt_colors[nt_name], s=20, alpha=1.0, zorder=2)
            fitx = np.logspace(0, 4, 200)
            fity = ((alpha_fit**gamma_fit)/(zeta(gamma_fit, 1 / alpha_fit) - alpha_fit**gamma_fit)) * (1.0 + alpha_fit * fitx) ** (-gamma_fit)
            ax.plot(fitx, fity, 'k-', linewidth=2, zorder=3)
            fit_rbrt = (gamma_fit - 1)/(gamma_fit - 2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1, 10**4)
        # Keeps all major tick marks every decade
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
        #ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
        # Custom Formatter: Labels only 10^0, 10^2, 10^4.
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: f"$10^{{{int(np.round(np.log10(x)))}}}$" if x > 0 and int(np.round(np.log10(x))) in {0, 2, 4} else ""
        ))
        
        ax.set_ylim(10**(-8), 10**0)
        # Keeps all major tick marks every decade
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
        #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
        # Custom Formatter: Labels only 10^-8, 10^-4, 10^0.
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: f"$10^{{{int(np.round(np.log10(y)))}}}$" if y > 0 and int(np.round(np.log10(y))) in {-8, -4, 0} else ""
        ))

        ax.tick_params(direction='out', which='minor', length=1.5, width=0.25)
        
        if " " in nt_name:
            words = nt_name.split(" ", 1)
            wrapped_name = f"{words[0]}\n{words[1]}"
        else:
            wrapped_name = nt_name
            
        # Box background face and edge alphas explicitly muted to force 100% transparency
        ax.text(0.95, 0.95, wrapped_name, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='none', edgecolor='none'))
        
        """
        # Box for fit rb/rt
        ax.text(0.05, 0.05, f'$r_b/r_t$\n$=${fit_rbrt:.2f}', transform=ax.transAxes, fontsize=5,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='none', edgecolor='none'))
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        """

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.savefig(os.path.join(OUT_FOLD, 'fig5_16_neurotransmitter_mosaic.svg'), format='svg')
    plt.close(fig_nt_mos)
else:
    print(f"Warning: Connection data not found at {CONN_DATA_CSV}. Skipping mosaic panels.")

# --- 3. Neuropil Mean Heterogeneity Multiline Plot ---
print("Generating Plot 17: Neuropil Multiline Profile...")
generate_category_multiline_plot(
    dataframe=df_np_clean,
    category_col='neuropil',
    active_categories=sorted_np_means,
    color_mapping=neuropil_colors,
    filename='fig5_17_neuropil_multiline.svg',
    target_size=SIZE_NP_MULTILINE,
    xlabel='Branches per connection'
)

# --- 4. Neurotransmitter Mean Heterogeneity Multiline Plot ---
print("Generating Plot 18: Neurotransmitter Multiline Profile...")
generate_category_multiline_plot(
    dataframe=df_nt_clean,
    category_col='neurotransmitter_type',
    active_categories=sorted_nt_means,
    color_mapping=nt_colors,
    filename='fig5_18_neurotransmitter_multiline.svg',
    target_size=SIZE_NT_MULTILINE,
    xlabel='Branches per connection'
)

print(f"\nSuccess! Scaled and formatted vectors compiled to target folder destination: {OUT_FOLD}")