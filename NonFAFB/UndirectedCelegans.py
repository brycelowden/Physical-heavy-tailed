CON_CSV = '[...]' # Raw C Elegans connectivity CSV
UND_CSV = '[...]' # Outputted undirected CSV

import pandas as pd
import numpy as np

print("Loading directed connectivity CSV...")

column_names = ['pre_id','post_id','synapse']
df = pd.read_csv(CON_CSV, header = None, names = column_names)

id_matrix = df[['pre_id','post_id']].values

sorted_ids = np.sort(id_matrix, axis = 1)

df['neuron_1'] = sorted_ids[:,0]
df['neuron_2'] = sorted_ids[:,1]

undirected_df = df.groupby(['neuron_1','neuron_2'], as_index=False)['synapse'].sum()

undirected_df.to_csv(UND_CSV, index = False)
