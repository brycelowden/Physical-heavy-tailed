import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

# Setup

MIN_CON = 25

CON_CSV    = '[...]' # Undirected FAFB connectivity CSV
SWC_FOLDER = '[...]' # Folder containing FAFB SWC neuron skeletons
OUTPUT_CSV = '[...]' # Output file 'Fig4_NeuronData.csv'

# Morphology

def analyze_single_swc(file_path):

    try:
        neuron_id = int(os.path.basename(file_path).replace('.swc', ''))
        
        columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=columns)

        parent_counts = df['parent'].value_counts()
        branch_points = (parent_counts[parent_counts.index != -1] > 1).sum()

        coords = df.set_index('id')[['x', 'y', 'z']]
        valid_links = df[df['parent'].isin(coords.index)].copy()
        
        child_coords = valid_links[['x', 'y', 'z']].values
        parent_coords = coords.loc[valid_links['parent']].values
        
        distances = np.linalg.norm(child_coords - parent_coords, axis=1)
        total_length_mm = np.sum(distances) / 1_000_000.0  # Convert from nm to mm

        return (neuron_id, int(branch_points), float(total_length_mm))
    except Exception as e:

        return None

# Heterogeneity

def calculate_het(array):
    s = np.sort(np.array(array, dtype=np.float64))
    n = s.shape[0]
    if n < 1 or np.sum(s) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * s) / (n * np.sum(s))

# Execution

if __name__ == "__main__":

    con_df = pd.read_csv(CON_CSV)
    
    strength_col = 'undirected_syn_count'
    id1_col, id2_col = con_df.columns[0], con_df.columns[1]

    # Map undirected edges to both neurons
    half_1 = con_df[[id1_col, strength_col]].rename(columns={id1_col: 'id', strength_col: 'strength'})
    half_2 = con_df[[id2_col, strength_col]].rename(columns={id2_col: 'id', strength_col: 'strength'})
    network_flat = pd.concat([half_1, half_2], ignore_index=True)
    
    network_flat = network_flat[network_flat['strength'] > 0]

    # Drop neurons with few connections
    connection_counts = network_flat['id'].value_counts()
    valid_neuron_ids = connection_counts[connection_counts >= MIN_CON].index
    
    network_flat = network_flat[network_flat['id'].isin(valid_neuron_ids)]
    
    grouped = network_flat.groupby('id')['strength']
    
    connectivity_results = []
    for neuron_id, strengths_series in tqdm(grouped, desc="Connectivity"):
        strengths = strengths_series.values
        
        total_cons = len(strengths)
        heterogeneity = calculate_het(strengths)
        
        connectivity_results.append({
            'id': neuron_id,
            'connections': total_cons,
            'heterogeneity': heterogeneity
        })
        
    connectivity_df = pd.DataFrame(connectivity_results)

    # SWC analysis
    
    swc_files = glob.glob(os.path.join(SWC_FOLDER, "*.swc"))
    
    morphology_results = []

    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(analyze_single_swc, swc_files), total=len(swc_files), desc="SWC Extraction"):
            if res is not None:
                morphology_results.append(res)
                
    morphology_df = pd.DataFrame(morphology_results, columns=['id', 'branches', 'length'])

    # only include neurons with connectivity and morphology data
    final_df = pd.merge(connectivity_df, morphology_df, on='id', how='inner')
    
    final_df['branches_per_length'] = final_df['branches'] / final_df['length']
    final_df['branches_per_connection'] = final_df['branches'] / final_df['connections']
    
    ordered_columns = [
        'id', 'branches', 'length', 'connections', 
        'branches_per_length', 'branches_per_connection', 
        'heterogeneity'
    ]
    final_df = final_df[ordered_columns]
    
    final_df.to_csv(OUTPUT_CSV, index=False)
