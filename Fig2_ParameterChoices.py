UND_CSV = '[...]'

import pandas as pd
import numpy as np

df = pd.read_csv(UND_CSV)

#keys: neuron IDs, values: appearance counts
conDict = dict()
syn_dict = dict()

print(f'\nCreating connectivity dictionary for {len(df)} unique connections')

#Count number of connections each neuronID appears in
for row in df.itertuples(index=False):
    conDict[row[0]] = conDict.get(row[0],0) + 1
    conDict[row[1]] = conDict.get(row[1],0) + 1

    syn_dict[row[0]] = syn_dict.get(row[0], 0) + row[2]
    syn_dict[row[1]] = syn_dict.get(row[1], 0) + row[2]

print('\nCreated connectivity dictionary')

connectionCounts = list(conDict.values())

print(f'\nAcross {len(conDict)} neurons, the average number of undirected connections per neuron is {np.mean(connectionCounts)}')

total_syns = df['undirected_syn_count'].sum()

print(f'\nThere are {total_syns} total synapses, for an average of {2 * total_syns / len(conDict)} synapses per neuron')

top_neuron = max(syn_dict, key=syn_dict.get)
top_synapses = syn_dict[top_neuron]

print(f"\nNeuron ID with the most synapses: {top_neuron}")
print(f"Total synapses for this neuron: {top_synapses}")
