import os
import re
import numpy as np
import pandas as pd
import bct  # brain connectivity toolbox for Python

# Set path to your connectome files
input_dir = "./connectomes"

def is_modularity_safe(matrix, min_nonzero=10, min_sum=1e-4):
    """Heuristic to decide if Louvain modularity is safe to run on this matrix."""
    # Check matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # Check for enough non-zero connections
    if np.count_nonzero(matrix) < min_nonzero:
        return False
    # Check for very low total weight (implies all edges are near-zero)
    if np.sum(matrix) < min_sum:
        return False
    return True


def extract_metrics(matrix, matrix_id=None):
    global_eff = np.nan
    char_path = np.nan
    mod = np.nan
    clustering = np.nan
    
    try:
        global_eff = bct.efficiency_wei(matrix)
    except Exception as e:
        print(f"Global efficiency failed on {matrix_id}: {e}")

    try:
        dist_matrix = bct.distance_wei(matrix)
        if isinstance(dist_matrix, tuple):
            dist_matrix = dist_matrix[0]
        char_path = bct.charpath(dist_matrix)[0]
    except Exception as e:
        print(f"Characteristic path length failed on {matrix_id}: {e}")

    try:
        if is_modularity_safe(matrix):
            mod, _ = bct.modularity_louvain_und(matrix)
        else:
            print(f"Modularity skipped on {matrix_id}: matrix not safe")
    except Exception as e:
        print(f"Modularity failed on {matrix_id}: {e}")

    try:
        clustering = np.mean(bct.clustering_coef_wu(matrix))
    except Exception as e:
        print(f"Clustering failed on {matrix_id}: {e}")

    return global_eff, char_path, mod, clustering



# Initialize list to hold results
results = []

# Process each file
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        # Extract subject and timepoint from filename
        match = re.match(r"sub-(\d+)_ses-(\d)\.txt", file)
        if match:
            subj = int(match.group(1))
            tp = int(match.group(2))

            # Read and clean matrix
            # Read raw file (tab- or space-separated)
            df_raw = pd.read_csv(os.path.join(input_dir, file), delimiter='\t', header=None, skiprows=2)

            # Drop the first 3 columns: 'data', 'data', node name
            df_clean = df_raw.iloc[:, 3:]

            # Now ensure the number of rows == number of columns
            matrix = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0).values

            # Check if square
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Matrix in {file} is not square: {matrix.shape}")


            # Extract metrics
            geff, cpl, mod, clust = extract_metrics(matrix, matrix_id=subj)

            # Append to results
            results.append({
                "subject": subj,
                "timepoint": tp,
                "global_efficiency": geff,
                "char_path_length": cpl,
                "modularity": mod,
                "clustering_coef": clust
            })


# Save to CSV
df = pd.DataFrame(results)
df.sort_values(by=["subject", "timepoint"], inplace=True)
df.to_csv("graph_metrics_long.csv", index=False)
print("Saved to graph_metrics_long.csv")
