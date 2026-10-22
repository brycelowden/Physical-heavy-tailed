import random as rnd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import time
from tqdm import tqdm
import numpy as np
import math
import pandas as pd
import igraph as ig
from collections import Counter, defaultdict
from scipy.special import zeta

OUT_SVG = '[...]'

print('Number of neurons?')
N = int(input('>'))

print('Max number of synapses?')
S = int(input('>'))

print('Branching rate?')
rb = float(input('>'))

print('Transition rate?')
rt = float(input('>'))

print('Synapse creation rate?')
rs = float(input('>'))

print('Number of trials per graph?')
trials = int(input('>'))

#Heterogeneity formula, given values and probabilities
def calculate_heterogeneity_prob(values, probabilities):
    
    v = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    sort_idx = np.argsort(v)
    v = v[sort_idx]
    p = p[sort_idx]

    expected_val = np.sum(p * v)

    cum_p = np.cumsum(p)

    # E[|s - s'|] = 2 * sum_{i=1}^n (p_i * v_i * (2 * P_i - p_i - 1))
    expected_abs_diff = 2 * np.sum(p * v * (2 * cum_p - p - 1))

    # H = 0.5 * (E[|s - s'|] / E[s])
    h = 0.5 * (expected_abs_diff / expected_val)

    return h

def run_simulation_gillespie(N, S, rb, rt, rs, trials):
    
    global_strength_counts = Counter()

    for k in range(trials):
        current_time = 0.0
        num_syns = 0

        occupants = {0: 1}
        strengths = defaultdict(int)

        while True:

            active_nodes = list(occupants.keys())
            active_counts = np.array([occupants[node] for node in active_nodes], dtype=int)
            
            node_branch_propensities = active_counts * rb
            node_trans_propensities = active_counts * rt
            node_syn_propensities = active_counts * rs
            
            total_b = node_branch_propensities.sum()
            total_t = node_trans_propensities.sum()
            total_s = node_syn_propensities.sum()
            R_total = total_b + total_t + total_s
                
            all_propensities = np.concatenate([
                node_branch_propensities, 
                node_trans_propensities, 
                node_syn_propensities
            ])
            
            r = rnd.uniform(0, R_total)
            cumulative_propensities = np.cumsum(all_propensities)
            event_idx = np.searchsorted(cumulative_propensities, r)
            
            num_nodes = len(active_nodes)
            phase_idx = event_idx // num_nodes
            node_idx = event_idx % num_nodes
            chosen_node = active_nodes[node_idx]
            
            # Branch
            if phase_idx == 0:
                occupants[chosen_node] += 1
            
            # Transition
            elif phase_idx == 1:
                occupants[chosen_node] -= 1
                
                #Discard nodes with no tips
                if occupants[chosen_node] <= 0:
                    del occupants[chosen_node]
                
                target = rnd.randint(0, N - 1)
                    
                occupants[target] = occupants.get(target, 0) + 1

            # Synapse    
            elif phase_idx == 2:
                
                strengths[chosen_node] += 1
                num_syns += 1

                if num_syns >= S:
                    break

        global_strength_counts.update(strengths.values())
        print(f"Done with trial {k+1}/{trials} | Synapses made: {num_syns}", end='\r')

    # Combined output
    unique_strengths = sorted(global_strength_counts.keys())
    total_elements = sum(global_strength_counts.values())
    
    probabilities = [global_strength_counts[s] / total_elements for s in unique_strengths]
    return np.array(unique_strengths), np.array(probabilities)

def run_simulation_random_gillespie(N, S, rb, rt, rs, p, trials):

    g = ig.Graph.Erdos_Renyi(n=N, p=p, directed=False)

    while not g.is_connected():
        g = ig.Graph.Erdos_Renyi(n=N, p=p, directed=False)

    neighbors_list = g.get_adjlist()
    degrees = np.array([len(neighbors) for neighbors in neighbors_list], dtype=np.int32)

    #Create pointers to section of list containing a neuron's neighbors
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(degrees)

    #Create neighbor list
    flattened_neighbors = np.empty(offsets[-1], dtype=np.int32)
    for i, neighbors in enumerate(neighbors_list):
        flattened_neighbors[offsets[i]:offsets[i+1]] = neighbors

    # Neuron n's degree is offsets[n+1] - offsets[n]

    print(f'\nConstructed random graph with {N} neurons and {p} of edges')

    global_strength_counts = Counter()
    branch_counts = list()
    connection_counts = list()

    for j in range(trials):
        num_syns = 0
        
        start = np.random.randint(0,N)
        occupants = {start: 1}
        strengths = defaultdict(int)

        while True:

            active_nodes = list(occupants.keys())
            active_counts = np.array([occupants[node] for node in active_nodes], dtype=int)
            
            node_branch_propensities = active_counts * rb
            node_trans_propensities = active_counts * rt
            node_syn_propensities = active_counts * rs
            
            total_b = node_branch_propensities.sum()
            total_t = node_trans_propensities.sum()
            total_s = node_syn_propensities.sum()
            R_total = total_b + total_t + total_s

            all_propensities = np.concatenate([
                node_branch_propensities, 
                node_trans_propensities, 
                node_syn_propensities
            ])
            
            r = rnd.uniform(0, R_total)
            cumulative_propensities = np.cumsum(all_propensities)
            event_idx = np.searchsorted(cumulative_propensities, r)
            
            num_nodes = len(active_nodes)
            phase_idx = event_idx // num_nodes
            node_idx = event_idx % num_nodes
            chosen_node = active_nodes[node_idx]
            
            # 4. Execute the single continuous-time event

            # Branch
            if phase_idx == 0:
                occupants[chosen_node] += 1
            
            # Transition
            elif phase_idx == 1:
                occupants[chosen_node] -= 1
                
                #Discard nodes with no tips
                if occupants[chosen_node] <= 0:
                    del occupants[chosen_node]
                
                #Choose random neighbor
                curr_offset = offsets[chosen_node]

                degree = offsets[chosen_node + 1] - curr_offset
                target = flattened_neighbors[curr_offset + np.random.randint(0,degree)]

                occupants[target] = occupants.get(target, 0) + 1

            # Synapse    
            elif phase_idx == 2:

                strengths[chosen_node] += 1
                num_syns += 1


                if num_syns >= S:
                    break

        global_strength_counts.update(strengths.values())

    # Combined output
    unique_strengths = sorted(global_strength_counts.keys())
    total_elements = sum(global_strength_counts.values())
    
    probabilities = [global_strength_counts[s] / total_elements for s in unique_strengths]

    return np.array(unique_strengths), np.array(probabilities)

# Execution

p_vals = [1,0.01,0.0001]

hets = list()
global_unique_strengths = list()
global_probabilities = list()

for it_p in tqdm(p_vals,desc = "Simulating on random graphs"):

    if it_p == 1:
        unique_strengths, probabilities = run_simulation_gillespie(N=N,S=S,rb=rb,rt=rt,rs=rs,trials=trials)
    else:
        unique_strengths, probabilities = run_simulation_random_gillespie(N=N,S=S,rb=rb,rt=rt,rs=rs,p=it_p,trials=trials)

    global_unique_strengths.append(unique_strengths)
    global_probabilities.append(probabilities)

    het = calculate_heterogeneity_prob(values=unique_strengths,probabilities=probabilities)
    hets.append(het)

#Plotting

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

fig, axs = plt.subplots(1,3,figsize=(108 * mm,36 * mm), sharey = True, constrained_layout=True)

for i, ax in enumerate(axs):
    ax.scatter(global_unique_strengths[i],global_probabilities[i],color = '#00A0CC', s=49/4, clip_on=False, zorder=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1,10**3)
    ax.set_ylim(0.5*10**(-5),1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.supxlabel('Synapses per connection', fontsize=10)
fig.supylabel('Probability', fontsize=10)

plt.savefig(OUT_SVG)

# For random graph plot in S2, use:

# N = 139255   S = 1105   rb = 0.25   rt = 0.1   rs = 1   trials = 1000   p_vals = [1,0.01,0.0001]
