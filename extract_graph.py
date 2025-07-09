import os
import re
import argparse
import logging
import json
import numpy as np
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import concurrent.futures
from datetime import datetime

try:
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
except ImportError:
    gpu_available = False
    gpu_count = 0

import multiprocessing

CONFIG = {
    "input_dir": "./connectomes_countn2",
    "min_nonzero": 10,
    "min_sum": 1e-4,
    "output_prefix": "graph_metrics",
    "output_dir": ".",
    "n_cpus": multiprocessing.cpu_count()
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

def sanity_check(matrix, min_nonzero=10, min_sum=1e-4):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    if np.any(matrix < 0):
        return False
    if not np.allclose(matrix, matrix.T):
        return False
    if np.count_nonzero(matrix) < min_nonzero:
        return False
    if np.sum(matrix) < min_sum:
        return False
    return True

def preprocess_matrix(matrix):
    matrix = np.nan_to_num(matrix, nan=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix

def characteristic_path_length(matrix):
    G = nx.from_numpy_array(matrix)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.average_shortest_path_length(G, weight='weight')

def modularity_louvain(matrix):
    G = nx.from_numpy_array(matrix)
    partition = community_louvain.best_partition(G, weight='weight')
    mod = community_louvain.modularity(partition, G, weight='weight')
    return mod, partition

def clustering_coefficient(matrix):
    G = nx.from_numpy_array(matrix)
    return nx.average_clustering(G, weight='weight')

def small_world_sigma(matrix, n_rand=10):
    """Calculate small-worldness sigma as (C/Crand)/(L/Lrand)"""
    G = nx.from_numpy_array(matrix)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        C = nx.average_clustering(G, weight='weight')
        L = nx.average_shortest_path_length(G, weight='weight')
        # Generate random graphs with same degree sequence
        Crand_list = []
        Lrand_list = []
        for _ in range(n_rand):
            # Use degree sequence for randomization
            deg_seq = [d for n, d in G.degree()]
            try:
                Gr = nx.configuration_model(deg_seq)
                Gr = nx.Graph(Gr)  # Remove parallel edges
                Gr.remove_edges_from(nx.selfloop_edges(Gr))
                if nx.is_connected(Gr):
                    Crand_list.append(nx.average_clustering(Gr))
                    Lrand_list.append(nx.average_shortest_path_length(Gr))
            except Exception:
                continue
        if Crand_list and Lrand_list:
            Crand = np.mean(Crand_list)
            Lrand = np.mean(Lrand_list)
            sigma = (C / Crand) / (L / Lrand) if Crand > 0 and Lrand > 0 else np.nan
        else:
            sigma = np.nan
    except Exception:
        sigma = np.nan
    return sigma

def global_metrics(matrix):
    G = nx.from_numpy_array(matrix)
    n = len(G.nodes)
    possible_edges = n * (n - 1) / 2
    actual_edges = np.count_nonzero(np.triu(matrix, 1))
    density = actual_edges / possible_edges if possible_edges > 0 else np.nan

    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = np.nan

    try:
        transitivity = nx.transitivity(G)
    except Exception:
        transitivity = np.nan

    try:
        geff = nx.global_efficiency(G)
    except Exception:
        geff = np.nan

    return density, assortativity, transitivity, geff

def nodal_metrics(matrix):
    G = nx.from_numpy_array(matrix)

    try:
        degree = dict(G.degree(weight='weight'))
    except Exception:
        degree = {node: np.nan for node in G.nodes}

    try:
        strength = dict(G.degree(weight='weight'))
    except Exception:
        strength = {node: np.nan for node in G.nodes}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception:
        eigenvector = {node: np.nan for node in G.nodes}

    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    except Exception:
        betweenness = {node: np.nan for node in G.nodes}

    try:
        clustering = nx.clustering(G, weight='weight')
    except Exception:
        clustering = {node: np.nan for node in G.nodes}

    try:
        local_eff = nx.local_efficiency(G)
        local_eff_dict = {node: local_eff for node in G.nodes}
    except Exception:
        local_eff_dict = {node: np.nan for node in G.nodes}

    nodal = {}
    for node in G.nodes:
        nodal[node] = {
            "degree": degree.get(node, np.nan),
            "strength": strength.get(node, np.nan),
            "eigenvector_centrality": eigenvector.get(node, np.nan),
            "betweenness": betweenness.get(node, np.nan),
            "local_efficiency": local_eff_dict.get(node, np.nan),
            "clustering": clustering.get(node, np.nan)
        }
    return nodal

def extract_metrics(matrix, matrix_id=None):
    matrix = preprocess_matrix(matrix)
    if not sanity_check(matrix, CONFIG["min_nonzero"], CONFIG["min_sum"]):
        logging.warning(f"Matrix {matrix_id} failed sanity checks.")
        return [np.nan]*8, {}

    try:
        cpl = characteristic_path_length(matrix)
    except Exception as e:
        logging.warning(f"Characteristic path length failed on {matrix_id}: {e}")
        cpl = np.nan

    try:
        mod, _ = modularity_louvain(matrix)
    except Exception as e:
        logging.warning(f"Modularity failed on {matrix_id}: {e}")
        mod = np.nan

    try:
        clust = clustering_coefficient(matrix)
    except Exception as e:
        logging.warning(f"Clustering failed on {matrix_id}: {e}")
        clust = np.nan

    try:
        density, assortativity, transitivity, geff = global_metrics(matrix)
    except Exception as e:
        logging.warning(f"Global metrics failed on {matrix_id}: {e}")
        density = assortativity = transitivity = geff = np.nan

    try:
        small_world = small_world_sigma(matrix)
    except Exception as e:
        logging.warning(f"Small-worldness failed on {matrix_id}: {e}")
        small_world = np.nan

    try:
        nodal = nodal_metrics(matrix)
    except Exception as e:
        logging.warning(f"Nodal metrics failed on {matrix_id}: {e}")
        nodal = {}

    return (geff, cpl, mod, clust, density, assortativity, transitivity, small_world), nodal

def detailed_matrix_check(matrix, matrix_id=None):
    errors = []
    if matrix.shape[0] != matrix.shape[1]:
        errors.append(f"Matrix is not square: shape={matrix.shape}")
    if not np.allclose(matrix, matrix.T):
        errors.append("Matrix is not symmetric.")
    if np.any(np.isnan(matrix)):
        errors.append("Matrix contains NaN values.")
    if np.any(np.isinf(matrix)):
        errors.append("Matrix contains Inf values.")
    if np.any(matrix < 0):
        errors.append("Matrix contains negative values.")
    if not np.all(np.diag(matrix) == 0):
        errors.append("Diagonal elements are not all zero.")
    if errors:
        logging.error(f"Matrix {matrix_id} invalid: {' | '.join(errors)}")
        return False
    return True

def process_file(args):
    file, input_dir = args
    match = re.match(r"sub-(\d+)_ses-(\d)\.txt", file)
    if not match:
        logging.warning(f"Filename {file} does not match expected pattern.")
        return None, None

    subj = int(match.group(1))
    tp = int(match.group(2))

    try:
        df_raw = pd.read_csv(os.path.join(input_dir, file), delimiter='\t', header=None, skiprows=2)
        df_clean = df_raw.iloc[:, 3:]
        matrix = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0).values
    except Exception as e:
        logging.error(f"Error reading {file}: {e}")
        return None, None

    # Shape-Check
    logging.info(f"Matrix {file}: shape={matrix.shape}")
    if matrix.shape[0] != matrix.shape[1]:
        logging.warning(f"Matrix {file} is not square: shape={matrix.shape}")

    matrix = preprocess_matrix(matrix)
    if not detailed_matrix_check(matrix, matrix_id=f"sub-{subj}_ses-{tp}"):
        return None, None

    logging.info(f"Processing sub-{subj}_ses-{tp} ...")
    metrics_global, metrics_nodal = extract_metrics(matrix, matrix_id=f"sub-{subj}_ses-{tp}")
    geff, cpl, mod, clust, density, assortativity, transitivity, small_world = metrics_global

    global_result = {
        "subject": subj,
        "timepoint": tp,
        "global_efficiency": geff,
        "char_path_length": cpl,
        "modularity": mod,
        "clustering_coef": clust,
        "density": density,
        "assortativity": assortativity,
        "transitivity": transitivity,
        "small_world": small_world
    }

    nodal_result = []
    for node, vals in metrics_nodal.items():
        nodal_result.append({
            "subject": subj,
            "timepoint": tp,
            "node": node,
            **vals
        })

    return global_result, nodal_result

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract graph metrics from connectome matrices.",
        epilog="Example: python extract_graph.py --config config.json",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default=None,
        help="Folder with .txt matrices (overrides config file)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Folder for output files (overrides config file)"
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=None,
        help="Number of CPUs to use (overrides config file)"
    )
    args = parser.parse_args()

    # Load config from JSON if given
    if args.config:
        with open(args.config, "r") as f:
            file_config = json.load(f)
        CONFIG.update({k: v for k, v in file_config.items() if v is not None})

    # Override config with CLI args if given
    if args.input_dir:
        CONFIG["input_dir"] = args.input_dir
    if args.output_dir:
        CONFIG["output_dir"] = args.output_dir
    if args.n_cpus:
        CONFIG["n_cpus"] = args.n_cpus

    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    n_cpus = CONFIG.get("n_cpus", multiprocessing.cpu_count())

    # CPU/GPU Info
    cpu_count = multiprocessing.cpu_count()
    if gpu_available:
        logging.info(f"Using GPU acceleration: {gpu_count} GPU(s) detected.")
    else:
        logging.info(f"Using CPU: {cpu_count} cores detected. Using {n_cpus} worker(s).")

    # Create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Output folder '{output_dir}' was created.")

    logging.info(f"Starting processing in folder: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    n_files = len(files)
    if not files:
        logging.error("No matching .txt files found.")
        return

    global_results = []
    nodal_results = []

    logging.info(f"Starting parallel processing of {n_files} files ...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(executor.map(process_file, [(file, input_dir) for file in sorted(files)]))

        for idx, (global_result, nodal_result) in enumerate(results, 1):
            if global_result is not None:
                global_results.append(global_result)
            if nodal_result is not None:
                nodal_results.extend(nodal_result)
            logging.info(f"Progress: {idx}/{n_files} files processed.")

        # Timestamp for output files
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_prefix = f"{CONFIG['output_prefix']}_{now}"
        global_path = os.path.join(output_dir, f"{out_prefix}_global.csv")
        nodal_path = os.path.join(output_dir, f"{out_prefix}_nodal.csv")

        pd.DataFrame(global_results).sort_values(by=["subject", "timepoint"]).to_csv(global_path, index=False)
        pd.DataFrame(nodal_results).sort_values(by=["subject", "timepoint", "node"]).to_csv(nodal_path, index=False)
        logging.info(f"✅ Done! {len(global_results)} global and {len(nodal_results)} nodal results saved as '{global_path}' and '{nodal_path}'.")
    finally:
        logging.info("Processing finished (finally).")

if __name__ == "__main__":
    main()
