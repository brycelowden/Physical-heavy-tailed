UND_CSV = '[...]' # Undirected proofread edges CSV

import pandas as pd
import numpy as np

df = pd.read_csv(UND_CSV)

#keys: neuron IDs, values: appearance counts
conDict = dict()

for row in df.itertuples(index=False):
    conDict[row[0]] = conDict.get(row[0],0) + 1
    conDict[row[1]] = conDict.get(row[1],0) + 1

print('\nCreated connectivity dictionary')

connectionCounts = list(conDict.values())

print(f'\nAcross {len(conDict)} neurons, the average number of undirected connections per neuron is {np.mean(connectionCounts)}')

print(f'\nThere are {df['synapses'].sum()} total synapses')
