import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
SELECTED_DIR = os.path.join(RESULTS_DIR, "selected_candidates")

os.makedirs(SELECTED_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'scores_with_stotal.txt'), sep='\t')

s_total_clean = df['S_total'].dropna()

print(df['S_total'].describe())

# count how many have S_total > 0.95
count_above_095 = (s_total_clean > 0.95).sum()
total_count = s_total_clean.count()
print(f"Number of entries with S_total > 0.95: {count_above_095} out of {total_count}")
df_screened = df[df['S_total'] > 0.95]
df_screened.to_csv(os.path.join(SELECTED_DIR, 'MP_95.txt'), sep='\t', index=False)

df_cleaned = df[df['S_total'].notna()]
df_cleaned.to_csv(os.path.join(DATA_DIR, 'MP_cleaned.txt'), sep='\t', index=False)

# check if those materials are stable
target_ids = df['Key'].tolist()
formulas = []
stabilities = []
energies_above_hull = []


'''
for i, mp_id in enumerate(target_ids):
    if i % 1000 == 0:
        print(f"Processing {i+1}/{len(target_ids)}")
    output_ind = np.nonzero(mat_names == mp_id)[0]
    if len(output_ind) == 0:
        formulas.append("N/A")
        stabilities.append("N/A")
        energies_above_hull.append("N/A")
    else:
        doc = docs[output_ind[0]]
        formulas.append(doc['formula_pretty'])
        stabilities.append(doc['is_stable'])
        energies_above_hull.append(doc['energy_above_hull'])

    
df['Formula'] = formulas
df['Is_Stable'] = stabilities
df['Energy_Above_Hull'] = energies_above_hull

df.to_csv('../data/scores_post.txt', sep='\t', index=False)
df_stable = df[df['Is_Stable'] == True]
df_screened = df_stable[df_stable['S_total'] > 0.95]
df_screened.to_csv('../data/scores_screened.txt', sep='\t', index=False)'''
