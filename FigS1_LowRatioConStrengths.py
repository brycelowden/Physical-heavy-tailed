import random as rnd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import numpy as np
import math
from collections import Counter, defaultdict
from scipy.special import zeta

print('Number of neurons?')
N = int(input('>'))

print('Max number of synapses?')
S = int(input('>'))

print('Branching rate?')
rb = float(input('>'))

print('Synapse creation rate?')
rs = float(input('>'))

print('Number of trials?')
trials = int(input('>'))

plotXvals = list()
plotYvals = list()

def theory(rb,rt,rs,s):
    alpha = (rb-rt)/rs
    gamma = 2 + 1/(rb/rt - 1)

    c = alpha**gamma/(zeta(gamma,1/alpha) - alpha**gamma)

    return c*(1+alpha * s)**(-1 * gamma)

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

NUM_PLOTS = 4

rts = [rb/1.1,rb/1.25,rb/1.5,rb/2]

unique_strengths=list()
probabilities=list()

for i in range(NUM_PLOTS):
    newstrengths, newprobs = run_simulation_gillespie(N, S, rb=rb, rt=rts[i], rs=rs, trials=trials)
    unique_strengths.append(newstrengths)
    probabilities.append(newprobs)

    print(f'\nDone with simulation {i+1}')

print('\nDone with simulations')

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

fig, axs = plt.subplots(1,NUM_PLOTS,figsize=(180 * mm,50 * mm),sharey=True, constrained_layout=True)

cmap = plt.colormaps['plasma']
colors = [cmap(1/8),cmap(3/8),cmap(5/8),cmap(7/8)]

for idx,ax in enumerate(axs):
    theoryXvals = np.logspace(np.log10(1),np.log10(10**4),1000)
    ax.scatter(unique_strengths[idx],probabilities[idx], color = colors[idx],s = 49/2,clip_on=False, zorder=10,label = 'Numeric')
    ax.plot(theoryXvals,theory(rb=rb,rt=rts[idx],rs=rs,s=theoryXvals),color = "#000000", linewidth = 2,solid_capstyle='round',clip_on=True,zorder=20,label = 'Analytic')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(1,10**3)
    ax.set_ylim(5*10**(-6),1)

axs[0].legend(frameon = False, loc='upper right')

fig.supxlabel('Synapses per connection', fontsize=10)
fig.supylabel('Probability', fontsize=10)

plt.savefig('[...]')

#For quad plot in figure S1,

#N = 139255   use S = 1105   rb = 0.25   rt = [rb/1.1,rb/1.25,rb/1.5,rb/2]   rs = 1   Trials = 1000

#colors = [cmap(1/8),cmap(3/8),cmap(5/8),cmap(7/8)]
