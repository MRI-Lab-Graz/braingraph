#!/usr/bin/env python3
"""
Calculate small-world properties for brain networks.
Small-world networks are particularly relevant for exercise studies.
"""

import pandas as pd
import numpy as np

def calculate_small_world_metrics(df):
    """
    Calculate small-world properties from global metrics.
    
    Small-world networks have:
    - High clustering (like regular networks)
    - Short path lengths (like random networks)
    """
    
    # Small-world coefficient (Humphries & Gurney, 2008)
    # SW = (C_real/C_random) / (L_real/L_random)
    # For brain networks, typically use theoretical values:
    # C_random ≈ density, L_random ≈ ln(N)/ln(density*N)
    
    df_sw = df.copy()
    
    # Estimate number of nodes (common brain parcellations)
    # You can adjust this based on your actual parcellation
    n_nodes = 100  # Adjust based on your atlas
    
    # Calculate normalized clustering
    df_sw['clustering_normalized'] = df_sw['clustering_coef'] / df_sw['density']
    
    # Calculate normalized path length
    # For fully connected networks: L_random ≈ ln(N)/ln(k) where k = density*N
    k = df_sw['density'] * n_nodes
    l_random = np.log(n_nodes) / np.log(k)
    df_sw['path_length_normalized'] = df_sw['char_path_length'] / l_random
    
    # Small-world coefficient
    df_sw['small_world_coef'] = df_sw['clustering_normalized'] / df_sw['path_length_normalized']
    
    # Alternative: Sigma coefficient (Humphries & Gurney, 2008)
    df_sw['sigma'] = (df_sw['clustering_coef'] / df_sw['density']) / df_sw['path_length_normalized']
    
    # Network efficiency balance
    df_sw['efficiency_balance'] = df_sw['global_efficiency'] / (1 - df_sw['modularity'])
    
    return df_sw[['subject', 'timepoint', 'small_world_coef', 'sigma', 'efficiency_balance']]

def add_small_world_to_global_metrics():
    """Add small-world metrics to the global metrics file."""
    
    # Load global metrics
    df = pd.read_csv('graph_metrics_global.csv')
    
    # Calculate small-world properties
    sw_metrics = calculate_small_world_metrics(df)
    
    # Merge with original data
    df_enhanced = df.merge(sw_metrics, on=['subject', 'timepoint'])
    
    # Save enhanced dataset
    df_enhanced.to_csv('graph_metrics_global_enhanced.csv', index=False)
    print("Enhanced global metrics saved to: graph_metrics_global_enhanced.csv")
    
    return df_enhanced

if __name__ == "__main__":
    df_enhanced = add_small_world_to_global_metrics()
    print(f"Added small-world metrics. New shape: {df_enhanced.shape}")
    print(f"New metrics: small_world_coef, sigma, efficiency_balance")
