import os
import pandas as pd

# Setup

UND_CON_CSV = '[...]' # FAFB Undirected connections CSV
MORPH_CSV   = '[...]' # Use 'Fig4_NeuronData.csv'
OUTPUT_CSV  = '[...]' # Outputted search results

# Search mode: 'single' to find individual neurons, 'pair' to find pairs of connected neurons
SEARCH_MODE = 'single'  

def filter_single_neuron(row):
    condition = (500 <= row['branches'] <= 1000) and (60 <= row['connections'] <= 100)
    return condition


def filter_neuron_pairs(row):
    total_branches = row['branches_1'] + row['branches_2']
    condition = (row['undirected_syn_count'] == 5 and total_branches <= 50)
    return condition


df_morph = pd.read_csv(MORPH_CSV)
df_morph['id'] = df_morph['id'].astype(str)

if SEARCH_MODE == 'single':
    
    matches = df_morph[df_morph.apply(filter_single_neuron, axis=1)].copy()
    
    output_cols = ['id', 'branches', 'connections', 'heterogeneity']
    final_output = matches[output_cols] if not matches.empty else pd.DataFrame(columns=['id'])
    
    print(f"Found {len(final_output)} single neurons.")

elif SEARCH_MODE == 'pair':

    df_conn = pd.read_csv(UND_CON_CSV)
    df_conn['neuron_id_1'] = df_conn['neuron_id_1'].astype(str)
    df_conn['neuron_id_2'] = df_conn['neuron_id_2'].astype(str)
    


    df_merged = pd.merge(df_conn, df_morph, left_on='neuron_id_1', right_on='id', how='inner')

    df_merged = pd.merge(df_merged, df_morph, left_on='neuron_id_2', right_on='id', how='inner', suffixes=('_1', '_2'))
    
    df_merged = df_merged.drop(columns=['id_1', 'id_2'], errors='ignore')
    
    matches = df_merged[df_merged.apply(filter_neuron_pairs, axis=1)].copy()
    
    output_cols = ['neuron_id_1', 'neuron_id_2', 'undirected_syn_count', 'branches_1', 'branches_2', 'heterogeneity_1', 'heterogeneity_2']
    final_output = matches[output_cols] if not matches.empty else pd.DataFrame(columns=['neuron_id_1', 'neuron_id_2'])
    
    print(f"Found {len(final_output)} pairs.")

else:
    raise ValueError("Invalid search mode")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_output.to_csv(OUTPUT_CSV, index=False)
