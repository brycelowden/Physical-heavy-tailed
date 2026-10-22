import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import graph-tool utilities
import graph_tool.all as gt

# =============================================================================
# 1. FILE PATH CONFIGURATIONS (NATIVE LINUX DIRECTORY)
# =============================================================================
BASE_DIR        = '/home/bryce/HeavyTailFiles'

CURRENT_FIG5    = f'{BASE_DIR}/Fig5_SortedNeurons.csv'
CON_CSV         = f'{BASE_DIR}/FlyWireDirCons.csv'
UPDATED_FIG5    = f'{BASE_DIR}/Fig5_SortedNeurons_Stage2.csv'

# =============================================================================
# 2. LOAD PREVIOUS FIGURE 5 DATASET
# =============================================================================
print("Loading current Fig5 sorting matrix...")
fig5_df = pd.read_csv(CURRENT_FIG5)
fig5_df['id'] = fig5_df['id'].astype(str)

target_ids = set(fig5_df['id'].unique())
print(f"Targeting metric layering for {len(target_ids)} index neurons.")

# =============================================================================
# 3. BUILD DIRECTED GRAPH ARCHITECTURE (DROPPING DUPLICATES)
# =============================================================================
print("Streaming directed connectivity tables and removing multi-synaptic redundancies...")
con_cols = ['pre_root_id', 'post_root_id']

unique_edges = set()
chunk_size = 100000
raw_chunks = pd.read_csv(CON_CSV, usecols=con_cols, chunksize=chunk_size, dtype={'pre_root_id': str, 'post_root_id': str})

for chunk in tqdm(raw_chunks, desc="Filtering Synaptic Splitting"):
    clean_chunk = chunk.drop_duplicates(subset=['pre_root_id', 'post_root_id'])
    edges_zip = zip(clean_chunk['pre_root_id'], clean_chunk['post_root_id'])
    unique_edges.update(edges_zip)

print(f"Found {len(unique_edges)} unique directed connections. Instantiating C++ graph...")

all_nodes = pd.Series(list({node for edge in unique_edges for node in edge})).unique()
node_to_idx = {str(int(node)): idx for idx, node in enumerate(all_nodes)}
idx_to_node = {idx: str(int(node)) for idx, node in enumerate(all_nodes)}

g = gt.Graph(directed=True)
g.add_vertex(len(all_nodes))

src_indices = [node_to_idx[edge[0]] for edge in unique_edges]
tgt_indices = [node_to_idx[edge[1]] for edge in unique_edges]
edge_matrix = np.column_stack((src_indices, tgt_indices))
g.add_edge_list(edge_matrix)

del unique_edges, src_indices, tgt_indices, edge_matrix

num_v = g.num_vertices()
print(f"Directed C++ graph built with {num_v} vertices and {g.num_edges()} edges.")

# =============================================================================
# 4. HIGH PERFORMANCE CALCULATIONS
# =============================================================================
print("Calculating PageRank vectors...")
vprop_pr = gt.pagerank(g)

print("Calculating K-Core Decomposition (K-Shell indices)...")
vprop_kcore = gt.kcore_decomposition(g)

print("Calculating HITS Hubs and Authorities (Optimized C++ engine)...")
# gt.hits() returns an eigenvalue, a hub property map, and an authority property map
_, vprop_hub, vprop_auth = gt.hits(g)

print("Extracting directional degree counts directly from memory buffers...")
in_degrees = g.get_in_degrees(g.get_vertices())
out_degrees = g.get_out_degrees(g.get_vertices())

print("Compiling calculations into update dataframe...")
network_results = []
for idx in tqdm(range(num_v), desc="Structuring Array Slices"):
    node_id = idx_to_node[idx]
    
    if node_id in target_ids:
        num_in = int(in_degrees[idx])
        num_out = int(out_degrees[idx])
        in_out_ratio = num_in / num_out if num_out > 0 else np.nan
        
        v = g.vertex(idx)
        
        network_results.append({
            'id': node_id,
            'num_in': num_in,
            'num_out': num_out,
            'in_out': in_out_ratio,
            'pagerank': float(vprop_pr[v]),
            'k_shell': int(vprop_kcore[v]),
            'hub_score': float(vprop_hub[v]),         # Added Hub metric
            'authority_score': float(vprop_auth[v])   # Added Authority metric
        })

df_new_metrics = pd.DataFrame(network_results)

# =============================================================================
# 5. MERGE AND SAVE UPDATED REGISTRY
# =============================================================================
print("Merging new architectural vectors into existing Figure 5 table...")
final_updated_df = fig5_df.merge(df_new_metrics, on='id', how='left')

fill_dict = {
    'num_in': 0, 'num_out': 0, 'in_out': 0.0, 
    'pagerank': 0.0, 'k_shell': 0, 
    'hub_score': 0.0, 'authority_score': 0.0
}
final_updated_df = final_updated_df.fillna(value=fill_dict)

print(f"Writing updated dataset matrix out to: {UPDATED_FIG5}")
final_updated_df.to_csv(UPDATED_FIG5, index=False)
print("Update pipeline successful! HITS analysis successfully appended.")