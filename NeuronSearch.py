import os
import pandas as pd

# =============================================================================
# 1. PARAMETERS & FILE CONFIGURATIONS
# =============================================================================
UND_CON_CSV = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Data/FlyWireUndCons.csv'
MORPH_CSV   = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Figure 4/Fig4_NeuronData.csv'
OUTPUT_CSV  = 'C:/Users/bryce/OneDrive/Desktop/Research/Heavy-Tailed CS/FAFB/Neuron search/SearchResults.csv'

# Define search mode: 'single' to find individual neurons, 'pair' to find pairs of connected neurons
SEARCH_MODE = 'single'  

# =============================================================================
# 2. DEFINE YOUR INJECTABLE FILTER FUNCTIONS HERE
# =============================================================================

def filter_single_neuron(row):
    """
    Define conditions for searching single neurons.
    Available columns from MORPH_CSV:
    row['id'], row['branches'], row['length'], row['connections'],
    row['branches_per_length'], row['branches_per_connection'], row['heterogeneity']
    """
    # Example: Search for a single neuron with branches between 100 and 200, and connections > 10
    condition = (500 <= row['branches'] <= 1000) and (60 <= row['connections'] <= 100)
    return condition


def filter_neuron_pairs(row):
    """
    Define conditions for searching neuron pairs.
    Available columns:
    - row['neuron_id_1'], row['neuron_id_2']
    - row['undirected_syn_count'] (from connection matrix)
    - Morph properties for neuron 1: row['branches_1'], row['length_1'], row['heterogeneity_1'], etc.
    - Morph properties for neuron 2: row['branches_2'], row['length_2'], row['heterogeneity_2'], etc.
    """
    # Example Request: Connected pair with EXACTLY 5 synapses where the SUM of their branches is LESS THAN 50
    total_branches = row['branches_1'] + row['branches_2']
    
    condition = (row['undirected_syn_count'] == 5 and total_branches <= 50)
    return condition


# =============================================================================
# 3. CORE PROCESSING ENGINE
# =============================================================================
print("Loading Morphological Database...")
df_morph = pd.read_csv(MORPH_CSV)

# Ensure ID is treated consistently as integers/strings without decimal float truncation
df_morph['id'] = df_morph['id'].astype(str)

if SEARCH_MODE == 'single':
    print("Executing Single Neuron Property Filtering...")
    
    # Apply filter function row by row
    matches = df_morph[df_morph.apply(filter_single_neuron, axis=1)].copy()
    
    # Isolate relevant sorting/identification columns for output
    output_cols = ['id', 'branches', 'connections', 'heterogeneity']
    final_output = matches[output_cols] if not matches.empty else pd.DataFrame(columns=['id'])
    
    print(f"Success! Found {len(final_output)} single neurons matching criteria.")

elif SEARCH_MODE == 'pair':
    print("Loading Undirected Connection Matrix...")
    df_conn = pd.read_csv(UND_CON_CSV)
    df_conn['neuron_id_1'] = df_conn['neuron_id_1'].astype(str)
    df_conn['neuron_id_2'] = df_conn['neuron_id_2'].astype(str)
    
    print("Mapping morphology properties to connection nodes...")
    # Map neuron 1 morphology attributes
    df_merged = pd.merge(df_conn, df_morph, left_on='neuron_id_1', right_on='id', how='inner')
    # Map neuron 2 morphology attributes (using suffixes to keep columns distinct)
    df_merged = pd.merge(df_merged, df_morph, left_on='neuron_id_2', right_on='id', how='inner', suffixes=('_1', '_2'))
    
    # Drop redundant identification columns left over from the merges
    df_merged = df_merged.drop(columns=['id_1', 'id_2'], errors='ignore')
    
    print("Executing Network Pair Repulsion & Property Filtering...")
    # Apply pair filter function row by row
    matches = df_merged[df_merged.apply(filter_neuron_pairs, axis=1)].copy()
    
    # Isolate key layout columns for cleaner results visualization
    output_cols = ['neuron_id_1', 'neuron_id_2', 'undirected_syn_count', 'branches_1', 'branches_2', 'heterogeneity_1', 'heterogeneity_2']
    final_output = matches[output_cols] if not matches.empty else pd.DataFrame(columns=['neuron_id_1', 'neuron_id_2'])
    
    print(f"Success! Found {len(final_output)} unique connectivity pairs matching criteria.")

else:
    raise ValueError("Invalid SEARCH_MODE configured. Choose either 'single' or 'pair'.")

# =============================================================================
# 4. SAVE EXPORT MATRIX
# =============================================================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_output.to_csv(OUTPUT_CSV, index=False)
print(f"Filtered database saved cleanly to destination layout matrix: {OUTPUT_CSV}")