import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

UND_CON_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireUndCons.csv'
DIR_CON_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireDirCons.csv'
OUT_SVG     = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 1/ConStrength.svg'

os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)

print("Loading undirected connection matrix...")
df_und = pd.read_csv(UND_CON_CSV)
syn_counts = df_und['undirected_syn_count'].values
neuron_ids = df_und[['neuron_id_1','neuron_id_2']].values
unique_ids = np.unique(neuron_ids)

num_connections = len(syn_counts)
total_synapses = np.sum(syn_counts)

Lambda = total_synapses / num_connections

print(f"Total neurons: {len(unique_ids)}")
print(f"Total synapses: {total_synapses}")
print(f"Total connections: {num_connections}")

def get_distribution_np(values, normalization_denom):

    active_values = values[values > 0]
    
    x, counts = np.unique(active_values, return_counts=True)

    y = counts / normalization_denom
    return x, y

print("Computing original connection strength distribution...")
x_real, y_real = get_distribution_np(syn_counts, num_connections)

print("Performing global shuffle...")

shuffled_global_bins = np.random.multinomial(total_synapses, [1.0 / num_connections] * num_connections)
x_global, y_global = get_distribution_np(shuffled_global_bins, num_connections)

print("Performing per-neuron shuffles...")

# Compute total strength and degree for each node

node_strengths = pd.concat([
    df_und.groupby('neuron_id_1')['undirected_syn_count'].sum(),
    df_und.groupby('neuron_id_2')['undirected_syn_count'].sum()
]).groupby(level=0).sum().to_dict()

node_degrees = pd.concat([
    df_und.groupby('neuron_id_1').size(),
    df_und.groupby('neuron_id_2').size()
]).groupby(level=0).sum().to_dict()

# Shuffle from each neuron's perspective

shuffled_node_views = {}
for node, total_node_synapses in node_strengths.items():
    deg = node_degrees[node]
    shuffled_node_views[node] = np.random.multinomial(total_node_synapses, [1.0 / deg] * deg)

# Combine

node_edge_counters = {node: 0 for node in node_strengths.keys()}
shuffled_averaged_strengths = np.zeros(num_connections)

for i, row in enumerate(df_und.itertuples(index=False)):
    n1, n2 = row.neuron_id_1, row.neuron_id_2
    
    idx1 = node_edge_counters[n1]
    val1 = shuffled_node_views[n1][idx1]
    node_edge_counters[n1] += 1
    
    idx2 = node_edge_counters[n2]
    val2 = shuffled_node_views[n2][idx2]
    node_edge_counters[n2] += 1
    
    exact_sum = val1 + val2
    base_val = exact_sum // 2
    remainder = exact_sum % 2
    
    # Round randomly
    if remainder == 1:
        shuffled_averaged_strengths[i] = base_val + np.random.choice([0, 1])
    else:
        shuffled_averaged_strengths[i] = base_val

x_avg_shuffle, y_avg_shuffle = get_distribution_np(shuffled_averaged_strengths, num_connections)

def Poisson(s):
    return (Lambda ** s) * (np.e ** (-1 * Lambda)) / (list(math.gamma(val + 1) for val in s))

poisxdata = np.linspace(1, 100, 1000)
poisydata = Poisson(poisxdata)

# Plotting

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

fig, ax = plt.subplots(figsize=(90 * mm, 90 * mm), constrained_layout=True)

ax.scatter(x_real, y_real, color="#00A0CC", s=49, zorder=40, clip_on=False, label='Real')

ax.scatter(x_avg_shuffle, y_avg_shuffle, color="#003fa5ff", s=49, zorder=30, clip_on=False, label='Neuron shuffle')

ax.scatter(x_global, y_global, color="#808080", s=49, zorder=20, clip_on=False, label='Global shuffle')

ax.plot(poisxdata, poisydata, color="#000000", linewidth=1, zorder=10, clip_on=True, label='Poisson')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Synapses per connection')
ax.set_ylabel('Probability')
ax.set_xlim(1, 5*10**3)
ax.set_ylim(3*10**(-8), 1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.legend(frameon=False, loc='upper right')
plt.savefig(OUT_SVG, format='svg')
plt.close(fig)
print("Saved figure")