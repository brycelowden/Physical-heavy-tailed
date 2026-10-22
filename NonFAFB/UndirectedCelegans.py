CON_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/C. Elegans Data/Celegans.csv'
UND_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/C. Elegans Data/Undirected_Celegans.csv'

import pandas as pd
import numpy as np

print("Loading directed connectivity CSV...")

column_names = ['pre_id','post_id','synapse']
df = pd.read_csv(CON_CSV, header = None, names = column_names)

print("Sorting neuron IDs to enforce consistency (neuron_id_1 < neuron_id_2)...")
# Extract the directed ID pairs as a raw NumPy matrix
id_matrix = df[['pre_id','post_id']].values

#Sort in numerical order horizontally for each row
sorted_ids = np.sort(id_matrix, axis = 1)

#Add sorted ids as 2 new columns

df['neuron_1'] = sorted_ids[:,0]
df['neuron_2'] = sorted_ids[:,1]

undirected_df = df.groupby(['neuron_1','neuron_2'], as_index=False)['synapse'].sum()

undirected_df.to_csv(UND_CSV, index = False)