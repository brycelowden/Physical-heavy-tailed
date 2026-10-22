CON_CSV = '[...]' # Directed BANC connections from FlyWire
UND_CSV = '[...]'

import pandas as pd
import numpy as np

print("Loading directed connectivity CSV...")
df = pd.read_csv(CON_CSV)

df = df[df['syn_count'] > 0]

print("Sorting neuron IDs...")

id_matrix = df[['pre_root_id', 'post_root_id']].values

sorted_ids = np.sort(id_matrix, axis=1)

df['neuron_id_1'] = sorted_ids[:, 0]
df['neuron_id_2'] = sorted_ids[:, 1]

undirected_df = (
    df.groupby(['neuron_id_1', 'neuron_id_2'])['syn_count']
    .sum()
    .reset_index(name='undirected_syn_count')
)

undirected_df.to_csv(UND_CSV, index=False)
