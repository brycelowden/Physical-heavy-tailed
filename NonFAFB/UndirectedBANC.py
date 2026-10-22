CON_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/BANCData/BANCconnections.csv'
UND_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/BANCData/BANCUndirectedConnections.csv'

import pandas as pd
import numpy as np

print("Loading directed connectivity CSV...")
df = pd.read_csv(CON_CSV)

# Ensure we are only looking at valid connections with positive synapse counts
df = df[df['syn_count'] > 0]

print("Sorting neuron IDs to enforce consistency (neuron_id_1 < neuron_id_2)...")
# Extract the directed ID pairs as a raw NumPy matrix for fast computation
id_matrix = df[['pre_root_id', 'post_root_id']].values

# Sorting along axis=1 forces the smaller ID to the left column and larger to the right
sorted_ids = np.sort(id_matrix, axis=1)

# Assign these back to new explicitly named columns
df['neuron_id_1'] = sorted_ids[:, 0]
df['neuron_id_2'] = sorted_ids[:, 1]

print("Aggregating reciprocal connections and summing synapse counts...")
# Group by the new undirected pairs and sum their synapse counts.
# This cleanly combines a->b and b->a into a single row.
undirected_df = (
    df.groupby(['neuron_id_1', 'neuron_id_2'])['syn_count']
    .sum()
    .reset_index(name='undirected_syn_count')
)

print(f"Saving undirected network ({len(undirected_df):,} unique edges) to disk...")
undirected_df.to_csv(UND_CSV, index=False)

print("\nConversion successful! Preview of the new undirected file:")
print(undirected_df.head(10))