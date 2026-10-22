UND_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/BANCData/BANCUndirectedConnections.csv'

import pandas as pd
import numpy as np

df = pd.read_csv(UND_CSV)

#keys: neuron IDs, values: appearance counts
conDict = dict()

print(f'\nCreating connectivity dictionary for {len(df)} unique connections')

#Count number of connections each neuronID appears in
for row in df.itertuples(index=False):
    conDict[row[0]] = conDict.get(row[0],0) + 1
    conDict[row[1]] = conDict.get(row[1],0) + 1

print('\nCreated connectivity dictionary')

connectionCounts = list(conDict.values())

print(f'\nAcross {len(conDict)} neurons, the average number of undirected connections per neuron is {np.mean(connectionCounts)}')

print(f'\nThere are {df['undirected_syn_count'].sum()} total synapses')