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

def hettheory(rb,rt,rs):
    alpha = (rb-rt)/rs
    gamma = 2 + 1/(rb/rt - 1)

    c = alpha**gamma/(zeta(gamma,1/alpha) - alpha**gamma)

    mean = c * alpha**(-gamma) * (zeta(gamma-1,1 + 1/alpha) - (1/alpha)*zeta(gamma,1 + 1/alpha))

    mad = 2 * mean - 2 * c**2 * alpha**(-2 * gamma) * sum((zeta(gamma, k + 1/alpha))**2 for k in range(1,50+1))

    return 0.5 * mad / mean

#Heterogeneity formula, given values and probabilities
def calculate_heterogeneity_prob(values, probabilities):
    
    v = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    # Sort values in ascending order and rearrange probabilities to match -> O(n log n)
    sort_idx = np.argsort(v)
    v = v[sort_idx]
    p = p[sort_idx]

    #Normalize probabilities

    # Calculate Expected Value E[s] (equivalent to the mean <s>)
    expected_val = np.sum(p * v)

    # Compute Cumulative Probabilities: P_i = sum_{j=1}^i p_j
    # np.cumsum provides [P_1, P_2, ..., P_n]
    cum_p = np.cumsum(p)

    # Calculate E[|s - s'|] using the cumulative weight sorting shortcut:
    # E[|s - s'|] = 2 * sum_{i=1}^n (p_i * v_i * (2 * P_i - p_i - 1))
    expected_abs_diff = 2 * np.sum(p * v * (2 * cum_p - p - 1))

    # H = 0.5 * (E[|s - s'|] / E[s])
    h = 0.5 * (expected_abs_diff / expected_val)

    return h

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

NUM_PLOTS = 3

rts = [rb/2,rb/3,rb/10]

unique_strengths=list()
probabilities=list()

for i in range(NUM_PLOTS):
    newstrengths, newprobs = run_simulation_gillespie(N, S, rb=rb, rt=rts[i], rs=rs, trials=trials)
    unique_strengths.append(newstrengths)
    probabilities.append(newprobs)

    print(f'\nDone with simulation {i+1}')

print('\nDone with simulations')

#Trimming outliers
maxPlotInd = list()

"""
for i in range(NUM_PLOTS):
    theoryXvals = np.logspace(np.log10(1),np.log10(10**4),1000)
    cutoff = 1
    while theory(rb=rb,rt=rts[i],rs=rs,s=cutoff)>=min(probabilities[i]):
        cutoff += 1
    
    cutoff *= 10

    ind = 0
    length = len(unique_strengths[i])

    while (ind < length - 1) & (unique_strengths[i][ind] < cutoff):
        ind += 1
    
    maxPlotInd.append(ind)
"""

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
plt.rcParams['legend.fontsize'] = 9      
plt.rcParams['legend.title_fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 7       
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

fig, axs = plt.subplots(1,NUM_PLOTS,figsize=(120 * mm,60 * mm),sharey=True, constrained_layout=True)

colors = ["#9c38e3ff",'#d96171ff','#f8cf34ff']

for idx,ax in enumerate(axs):
    theoryXvals = np.logspace(np.log10(1),np.log10(10**4),1000)
    #ax.scatter(unique_strengths[idx][:maxPlotInd[idx]],probabilities[idx][:maxPlotInd[idx]], color = colors[idx],s = 49/2,clip_on=False, zorder=10,label = 'Numeric')
    ax.scatter(unique_strengths[idx],probabilities[idx], color = colors[idx],s = 49/2,clip_on=False, zorder=10,label = 'Numeric')
    ax.plot(theoryXvals,theory(rb=rb,rt=rts[idx],rs=rs,s=theoryXvals),color = "#000000", linewidth = 2,solid_capstyle='round',clip_on=True,zorder=20,label = 'Analytic')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(1,2*10**3)
    ax.set_ylim(0.5*10**(-5),1)

axs[0].legend(frameon = False, loc='upper right')

fig.supxlabel('Synapses per connection', fontsize=10)
fig.supylabel('Probability', fontsize=10)

plt.savefig('C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/Figure Files/Figure 2/TripleConStrength_MaxSyn.svg')

#For triple plot in figure 2, use the following values:

#N = 139255   S = 1105   rb = 0.25   rt = [rb/2, rb/3, rb/10]   rs = 1   Trials = 1000

#colors = ['#00ccffff','#00cc88ff','#00cc44ff']