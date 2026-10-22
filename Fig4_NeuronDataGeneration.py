import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

# =============================================================================
# 1. PARAMETERS & CONFIGURATION
# =============================================================================
MIN_CON = 25

CON_CSV    = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireUndCons.csv'
SWC_FOLDER = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireSkeletons'
OUTPUT_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure 4/Fig4_NeuronData.csv'


# =============================================================================
# 2. MORPHOLOGY ANALYSIS PIPELINE (From LengthAndBranchData.py)
# =============================================================================
def analyze_single_swc(file_path):
    """
    Extracts (id, branch_points, total_length) from a single SWC file.
    Converts length directly from nanometers to millimeters.
    """
    try:
        neuron_id = int(os.path.basename(file_path).replace('.swc', ''))
        
        # Load SWC, skipping comments
        columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=columns)

        # 1. Calculate Branch Points
        parent_counts = df['parent'].value_counts()
        branch_points = (parent_counts[parent_counts.index != -1] > 1).sum()

        # 2. Calculate Total Length (Vectorized)
        coords = df.set_index('id')[['x', 'y', 'z']]
        valid_links = df[df['parent'].isin(coords.index)].copy()
        
        child_coords = valid_links[['x', 'y', 'z']].values
        parent_coords = coords.loc[valid_links['parent']].values
        
        # Vectorized Euclidean distance calculation (nanometers)
        distances = np.linalg.norm(child_coords - parent_coords, axis=1)
        total_length_mm = np.sum(distances) / 1_000_000.0  # Safe conversion to mm here

        return (neuron_id, int(branch_points), float(total_length_mm))
    except Exception as e:
        # Avoid crashing parallel execution on a corrupted SWC file
        return None


# =============================================================================
# 3. STATISTICAL ANALYSIS & CURVE FITTING (Log-Cauchy Optimization)
# =============================================================================
def calculate_gini(array):
    """
    Efficiently calculates the Gini coefficient (Heterogeneity Score).
    """
    s = np.sort(np.array(array, dtype=np.float64))
    n = s.shape[0]
    if n < 1 or np.sum(s) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * s) / (n * np.sum(s))

# =============================================================================
# 4. EXECUTION CONTROLLER
# =============================================================================
if __name__ == "__main__":
    print("--- STEP 1: Processing Connectivity Data ---")
    # Load undirected connection pairs
    con_df = pd.read_csv(CON_CSV)
    
    # Enforce confirmed structural names
    strength_col = 'undirected_syn_count'
    id1_col, id2_col = con_df.columns[0], con_df.columns[1]

    print("Duplicating undirected edges across both partner reference frames...")
    # Map undirected edges to BOTH participating node IDs so we see every neuron's entire field
    half_1 = con_df[[id1_col, strength_col]].rename(columns={id1_col: 'id', strength_col: 'strength'})
    half_2 = con_df[[id2_col, strength_col]].rename(columns={id2_col: 'id', strength_col: 'strength'})
    network_flat = pd.concat([half_1, half_2], ignore_index=True)
    
    # Filter out non-positive synapse entries to maintain mathematical stability
    network_flat = network_flat[network_flat['strength'] > 0]

    print(f"Applying MIN_CON threshold (>= {MIN_CON}) early to drop weak networks...")
    # Count connections per unique neuron id
    connection_counts = network_flat['id'].value_counts()
    valid_neuron_ids = connection_counts[connection_counts >= MIN_CON].index
    
    # Truncate the network down completely to save memory during aggregation loop
    network_flat = network_flat[network_flat['id'].isin(valid_neuron_ids)]
    
    print("Aggregating localized connectivity statistics and executing heavy-tailed curve fits...")
    # Group by neuron ID and isolate lists of all connection strengths
    grouped = network_flat.groupby('id')['strength']
    
    connectivity_results = []
    for neuron_id, strengths_series in tqdm(grouped, desc="Connections and Heterogeneity"):
        strengths = strengths_series.values
        
        # Calculate structural properties
        total_cons = len(strengths)
        heterogeneity = calculate_gini(strengths)
        
        connectivity_results.append({
            'id': neuron_id,
            'connections': total_cons,
            'heterogeneity': heterogeneity
        })
        
    connectivity_df = pd.DataFrame(connectivity_results)
    print(f"Finished connectivity matrix. Profiles generated for {len(connectivity_df)} neurons.")

    print("\n--- STEP 2: Processing Morphology Skeleton Files ---")
    swc_files = glob.glob(os.path.join(SWC_FOLDER, "*.swc"))
    print(f"Discovered {len(swc_files):,} morphology files. Initializing multi-core workers...")
    
    morphology_results = []
    # Utilize full physical CPU cores via process pooling
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(analyze_single_swc, swc_files), total=len(swc_files), desc="SWC Extraction"):
            if res is not None:
                morphology_results.append(res)
                
    morphology_df = pd.DataFrame(morphology_results, columns=['id', 'branches', 'length'])
    print(f"Morphology processing finished. Extracted {len(morphology_df)} valid skeletons.")

    print("\n--- STEP 3: Executing Inner Join Merger & Column Derivatives ---")
    # Execute structural inner join (ensures mutual intersection between morphology and network data)
    final_df = pd.merge(connectivity_df, morphology_df, on='id', how='inner')
    
    print("Calculating final analytical derivative ratios...")
    final_df['branches_per_length'] = final_df['branches'] / final_df['length']
    final_df['branches_per_connection'] = final_df['branches'] / final_df['connections']
    
    # Enforce precise column schema layout requested
    ordered_columns = [
        'id', 'branches', 'length', 'connections', 
        'branches_per_length', 'branches_per_connection', 
        'heterogeneity'
    ]
    final_df = final_df[ordered_columns]
    
    print(f"Saving compiled analysis to destination path...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Final consolidated records saved to: {OUTPUT_CSV}")