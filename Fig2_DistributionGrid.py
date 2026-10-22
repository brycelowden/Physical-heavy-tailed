import random as rnd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import time
import numpy as np
import math
from collections import Counter, defaultdict
from scipy.special import zeta

print('Number of neurons?')
N = int(input('>'))

print('Max number of synapses?')
S = int(input('>'))

print('Synapse creation rate?')
rs = float(input('>'))

print('Number of trials?')
trials = int(input('>'))

def run_simulation_gillespie(N, S, rb, rt, rs, trials):
    """
    Simulates the branching random walk using the continuous-time 
    Gillespie Algorithm until exactly S unique connections are established.
    """

    global_strength_counts = Counter()

    for k in range(trials):
        current_time = 0.0
        num_syns = 0
        
        # State tracking: { node_index: count }
        # Every trial begins with exactly 1 walker at a root node (node 0)
        occupants = {0: 1}
        strengths = defaultdict(int)

        while True:

            # 1. Calculate the active propensities per node
            active_nodes = list(occupants.keys())
            active_counts = np.array([occupants[node] for node in active_nodes], dtype=int)
            
            node_branch_propensities = active_counts * rb
            node_trans_propensities = active_counts * rt
            node_syn_propensities = active_counts * rs
            
            # Global propensity sums
            total_b = node_branch_propensities.sum()
            total_t = node_trans_propensities.sum()
            total_s = node_syn_propensities.sum()
            R_total = total_b + total_t + total_s
                
            # 3. Determine WHICH type of event happens and WHERE it happens
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
                
                #Discard nodes with no tips to avoid clutter
                if occupants[chosen_node] <= 0:
                    del occupants[chosen_node]
                
                target = rnd.randint(0, N - 1)
                    
                occupants[target] = occupants.get(target, 0) + 1

            # Synapse    
            elif phase_idx == 2:
                # Check if this node is gaining its very first synapse
                # Searching for chosen_node initializes it with value 0

                strengths[chosen_node] += 1
                num_syns += 1

                # Stop the trial immediately the moment our connection goal is met
                if num_syns >= S:
                    break

        # Save all non-zero synapse values from this trial run
        global_strength_counts.update(strengths.values())
        print(f"Done with trial {k+1}/{trials} | Synapses made: {num_syns}", end='\r')

    # Combined output
    unique_strengths = sorted(global_strength_counts.keys())
    total_elements = sum(global_strength_counts.values())
    
    probabilities = [global_strength_counts[s] / total_elements for s in unique_strengths]
    return np.array(unique_strengths), np.array(probabilities)

rbs = [1,1/5,1/25]
rts = [1/25,1/5,1]

par_pairs = [{'rb':rb,'rt':rt} for rb in rbs for rt in rts]

all_unique_strengths=list()
all_probabilities=list()

for i,par_pair in enumerate(par_pairs):
    newstrengths, newprobs = run_simulation_gillespie(N, S, rb=par_pair['rb'], rt=par_pair['rt'], rs=rs, trials=trials)
    
    all_unique_strengths.append(newstrengths)
    all_probabilities.append(newprobs)

    print(f'\nDone with simulation {i+1} which had rb = {par_pair['rb']} and rt = {par_pair['rt']}')

print('\nDone with simulations')

# Plot of connection strength versus probability
# 1. Define the millimeter conversion factor
mm = 1 / 25.4  # 1 mm in inches

# 2. Set global style
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

# (105mm, 90mm)
fig, axs = plt.subplots(3,3,figsize=(105 * mm,90 * mm),sharey=True, sharex = True, constrained_layout=True)

flat_axs = axs.flatten()

fig.supxlabel('Synapses per connection', fontsize=10)
fig.supylabel('Probability', fontsize=10)

colors = [
    (0,0.8,0.4), (0,0.6,0.4), (0,0.4,0.4),
    (0,0.8,0.6), (0,0.6,0.6), (0,0.4,0.6),
    (0,0.8,0.8), (0,0.6,0.8), (0,0.4,0.8)
]

for idx,ax in enumerate(flat_axs):
    ax.set_xlim(1,2*10**3)
    ax.set_ylim(10**(-6),1)

    ax.scatter(all_unique_strengths[idx],all_probabilities[idx], color = colors[idx],s = 49/2,clip_on=False, zorder=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #Tick handling
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
    #ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
        # Label only 1 and 10**3
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.round(np.log10(x)))}}}$" if x > 0 and int(np.round(np.log10(x))) in {0, 1, 2, 3} else ""))
    
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
    # Label only 1, 10**-3, 10**-6
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"$10^{{{int(np.round(np.log10(y)))}}}$" if y > 0 and int(np.round(np.log10(y))) in {-6, -3, 0} else ""))

plt.savefig('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 2/DistributionGrid.svg')

#For grid in figure 2, use the following values

#N = 139255   S = 1105   rs = 1   Trials = 1000

#colors = ['#00ccffff','#00cc88ff','#00cc44ff']