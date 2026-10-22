import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# This file must be run on a Linux (virtual) machine
# Generates the following neuron-level quantities: 
# 'neuropil','reciprocity','neurotransmitter_type','distance_olfactory','distance_optic','clustering_coefficient','closeness_centrality','betweenness_centrality'

import graph_tool.all as gt

BASE_DIR    = '[...]'

FIG4_CSV    = f'{BASE_DIR}/Fig4_NeuronData.csv'
CON_CSV     = f'{BASE_DIR}/FlyWireDirCons.csv'
UND_CSV     = f'{BASE_DIR}/FlyWireUndCons.csv'
PER_PARQUET = f'{BASE_DIR}/periphery_data.parquet'

OUT_CSV     = f'{BASE_DIR}/Fig5_SortedNeurons.csv'

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

REGION_MAPPING = {
    "Optic_Lobe": ["AME_L", "LA_L", "ME_L", "LO_L", "LOP_L","AME_R", "LA_R", "ME_R", "LO_R", "LOP_R"],
    "Lateral_Complex": ["BU_L","LAL_L","GA_L","BU_R","LAL_R","GA_R"],
    "Lateral_Horn": ["LH_L","LH_R"],
    "Periesophageal_NP": ["SAD","PRW","CAN_L","AMMC_L","FLA_L","CAN_R","AMMC_R","FLA_R"],
    "Inferior_NP": ["ICL_L","IB_L","ATL_L","CRE_L","SCL_L","ICL_R","IB_R","ATL_R","CRE_R","SCL_R"],
    "Ventromedial_NP": ["VES_L","GOR_L","SPS_L","IPS_L","EPA_L","VES_R","GOR_R","SPS_R","IPS_R","EPA_R"],
    "Mushroom_Body": ["MB_CA_L", "MB_CA_R", "MB_PED_L", "MB_PED_R","MB_VL_L","MB_VL_R","MB_ML_L","MB_ML_R"],
    "Antennal_Lobe": ["AL_L","AL_R"],
    "Superior_NP": ["SLP_L","SIP_L","SMP_L","SLP_R","SIP_R","SMP_R"],
    "Ventrolateral_NP": ["AVLP_L","PVLP_L","WED_L","PLP_L","AOTU_L","AVLP_R","PVLP_R","WED_R","PLP_R","AOTU_R"],
    "Central_Complex": ["NO","PB","EB","FB"],
    "Gnathal_Ganglia": ["GNG"],
    "Ocelli": ["OCG"],
    "Other": ["UNASGD"]
}

SUB_TO_TRUE_NP = {}
for true_np, sub_nps in REGION_MAPPING.items():
    for sub in sub_nps:
        SUB_TO_TRUE_NP[sub] = true_np

print("Loading Fig4 table...")
fig4_df = pd.read_csv(FIG4_CSV)

if 'id' in fig4_df.columns:
    fig4_df = fig4_df.rename(columns={'id': 'root_id'})
    fig4_df['root_id'] = fig4_df['root_id'].astype(str)
else:
    raise KeyError("Fig4 CSV file improperly setup")

target_neurons = set(fig4_df['root_id'].unique())

# Connectivity results

print("Processing Neuropil, Neurotransmitter, and Reciprocity...")
con_cols = ['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']

neuron_links = {nid: {'in': set(), 'out': set()} for nid in target_neurons}
neuron_np_synapses = defaultdict(lambda: defaultdict(int))
neuron_nt_synapses = defaultdict(lambda: defaultdict(int))

chunk_size = 100000
raw_chunks = pd.read_csv(CON_CSV, usecols=con_cols, chunksize=chunk_size, dtype={'pre_root_id': str, 'post_root_id': str})

for chunk in tqdm(raw_chunks, desc="Streaming Connectivity Columns"):
    chunk['true_np'] = chunk['neuropil'].map(SUB_TO_TRUE_NP).fillna('Other')
    
    pre_mask = chunk['pre_root_id'].isin(target_neurons)
    if pre_mask.any():
        for row in chunk[pre_mask].itertuples(index=False):
            syn = row.syn_count
            neuron_np_synapses[row.pre_root_id][row.true_np] += syn
            if pd.notna(row.nt_type):
                neuron_nt_synapses[row.pre_root_id][row.nt_type] += syn
            neuron_links[row.pre_root_id]['out'].add(row.post_root_id)
            
    post_mask = chunk['post_root_id'].isin(target_neurons)
    if post_mask.any():
        for row in chunk[post_mask].itertuples(index=False):
            syn = row.syn_count
            neuron_np_synapses[row.post_root_id][row.true_np] += syn
            if pd.notna(row.nt_type):
                neuron_nt_synapses[row.post_root_id][row.nt_type] += syn
            neuron_links[row.post_root_id]['in'].add(row.pre_root_id)

connectivity_results = []
for nid in tqdm(target_neurons):
    # Assign neuropil by plurality of synapses
    np_dict = neuron_np_synapses[nid]
    assigned_neuropil = max(np_dict, key=np_dict.get) if np_dict else np.nan
    
    # Assign neurotransmitter by plurality of synapses
    nt_dict = neuron_nt_synapses[nid]
    assigned_nt = max(nt_dict, key=nt_dict.get) if nt_dict else np.nan
    
    in_neighbors = neuron_links[nid]['in']
    out_neighbors = neuron_links[nid]['out']
    union_neighbors = in_neighbors.union(out_neighbors)
    mutual_neighbors = in_neighbors.intersection(out_neighbors)
    
    reciprocity_val = len(mutual_neighbors) / len(union_neighbors) if len(union_neighbors) > 0 else np.nan
        
    connectivity_results.append({
        'root_id': nid,
        'assigned_neuropil': assigned_neuropil,
        'neurotransmitter_type': assigned_nt,
        'reciprocity': reciprocity_val
    })

df_connectivity = pd.DataFrame(connectivity_results)

del neuron_links, neuron_np_synapses, neuron_nt_synapses

# Load peripheral data from parquet file

df_per = pd.read_parquet(PER_PARQUET, columns=["root_id", "distance_optic", "distance_olfactory"])
df_per['root_id'] = df_per['root_id'].astype(str)

# Graph quantities

df_und = pd.read_csv(UND_CSV, usecols=['neuron_id_1', 'neuron_id_2']).dropna()

all_nodes = pd.concat([df_und['neuron_id_1'], df_und['neuron_id_2']]).unique()
node_to_idx = {str(int(node)): idx for idx, node in enumerate(all_nodes)}
idx_to_node = {idx: str(int(node)) for idx, node in enumerate(all_nodes)}

g = gt.Graph(directed=False)
g.add_vertex(len(all_nodes))

src_indices = df_und['neuron_id_1'].astype(int).astype(str).map(node_to_idx).values
tgt_indices = df_und['neuron_id_2'].astype(int).astype(str).map(node_to_idx).values
edge_matrix = np.column_stack((src_indices, tgt_indices))
g.add_edge_list(edge_matrix)

del df_und, src_indices, tgt_indices, edge_matrix

num_v = g.num_vertices()
print(f"graph made with {num_v} vertices and {g.num_edges()} edges.")

print("Calculating clustering...")
vprop_clustering = gt.local_clustering(g)

print("Calculating approximate closeness centrality using landmarks...")
np.random.seed(42)
landmark_indices = np.random.choice(num_v, size=min(1000, num_v), replace=False)

sum_distances = np.zeros(num_v, dtype=np.float64)
max_possible_dist = num_v

for lm_idx in tqdm(landmark_indices, desc="Processing Closeness Landmarks"):
    single_dist_map = gt.shortest_distance(g, source=g.vertex(lm_idx))
    single_dist_array = single_dist_map.get_array()
    single_dist_array[single_dist_array > max_possible_dist] = max_possible_dist
    sum_distances += single_dist_array

closeness_array = np.zeros(num_v)
valid_mask = sum_distances > 0
closeness_array[valid_mask] = (len(landmark_indices) - 1) / sum_distances[valid_mask]

del sum_distances

print("Calculating approximate betweenness centrality...")

vprop_betweenness = g.new_vertex_property("double")

for pivot_idx in tqdm(landmark_indices):
    single_bv, _ = gt.betweenness(g, pivots=[pivot_idx])
    
    vprop_betweenness.a += single_bv.a
    
graph_results = []
for idx in tqdm(range(num_v)):
    v = g.vertex(idx)
    graph_results.append({
        'root_id': idx_to_node[idx],
        'clustering_coefficient': float(vprop_clustering[v]),
        'closeness_centrality': float(closeness_array[idx]),
        'betweenness_centrality': float(vprop_betweenness[v])
    })
df_graph_metrics = pd.DataFrame(graph_results)

# Combine data

final_df = fig4_df.copy()
final_df = final_df.merge(df_connectivity,  on='root_id', how='left')
final_df = final_df.merge(df_per,           on='root_id', how='left')
final_df = final_df.merge(df_graph_metrics, on='root_id', how='left')

graph_cols = ['clustering_coefficient', 'closeness_centrality', 'betweenness_centrality']
final_df[graph_cols] = final_df[graph_cols].fillna(0.0)

final_df = final_df.rename(columns={'root_id': 'id', 'assigned_neuropil': 'neuropil'})

desired_columns = [
    'id', 
    'branches_per_connection', 
    'heterogeneity', 
    'neuropil', 
    'reciprocity', 
    'neurotransmitter_type', 
    'distance_olfactory', 
    'distance_optic',
    'clustering_coefficient',
    'closeness_centrality',
    'betweenness_centrality'
]

final_df = final_df[[col for col in desired_columns if col in final_df.columns]]

final_df.to_csv(OUT_CSV, index=False)
