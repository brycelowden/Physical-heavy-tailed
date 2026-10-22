import random as rnd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from tqdm import tqdm
import numpy as np
import math
import pandas as pd
from collections import Counter, defaultdict
from scipy.special import zeta

OUT_CSV = '[...]'

print('Number of neurons?')
N = int(input('>'))

print('Max number of synapses?')
S = int(input('>'))

print('Min branching rate?')
min_rb = float(input('>'))

print('Max branching rate?')
max_rb = float(input('>'))

print('Min transition rate?')
min_rt = float(input('>'))

print('Max transition rate?')
max_rt = float(input('>'))

print('Synapse creation rate?')
rs = float(input('>'))

print('Resolution?')
res = int(input('>'))


print('Number of trials per simulation?')
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
    branch_counts = list()
    connection_counts = list()

    for k in range(trials):
        current_time = 0.0
        num_syns = 0
        
        # State tracking: { node_index: count }
        # Every trial begins with exactly 1 walker at a root node (node 0)
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

        # Each tip besides the first represents a branch event
        branch_counts.append(sum(occupants.values())-1)

        # Each node with at least one synapse is a connection
        connection_counts.append(len(strengths))

    # Combined output
    unique_strengths = sorted(global_strength_counts.keys())
    total_elements = sum(global_strength_counts.values())
    
    probabilities = [global_strength_counts[s] / total_elements for s in unique_strengths]

    mean_branches = np.mean(branch_counts)
    mean_connections = np.mean(connection_counts)

    return np.array(unique_strengths), np.array(probabilities), mean_branches, mean_connections

# Execution

rb_vals = np.logspace(np.log10(min_rb),np.log10(max_rb),res)
rt_vals = np.logspace(np.log10(min_rt),np.log10(max_rt),res)

par_pairs = list((rb,rt) for rb in rb_vals for rt in rt_vals)

sweep_results = list()

for pair in tqdm(par_pairs,desc = "Running parameter sweep"):
    
    unique_strengths, probabilities, branches, connections = run_simulation_gillespie(N=N,S=S,rb=pair[0],rt=pair[1],rs=rs,trials=trials)
    
    het = calculate_heterogeneity_prob(values=unique_strengths,probabilities=probabilities)

    row_data = {
        'rb':pair[0],
        'rt':pair[1],
        'rs':rs,
        'N':N,
        'S':S,
        'het':het,
        'branches':branches,
        'connections':connections
    }

    sweep_results.append(row_data)

results_df = pd.DataFrame(sweep_results)

results_df.to_csv(OUT_CSV, index=False)


# For triple plot in figure 2, use the following values:

# N = 139,255   S=1,105    trials = 100

# range both rb and rt from 0.025 to 2.5
