#!/usr/bin/env python3
"""
Network-specific analysis strategies for exercise intervention studies.
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

def rich_club_analysis(df):
    """
    Rich club analysis - highly connected nodes tend to connect to each other.
    Exercise may affect rich club organization.
    """
    
    # This requires connectivity matrices, but we can estimate from nodal metrics
    # Placeholder for rich club coefficient calculation
    
    rich_club_results = {}
    
    for timepoint in df['timepoint'].unique():
        tp_data = df[df['timepoint'] == timepoint]
        
        # Identify rich nodes (top 20% by degree)
        degree_threshold = tp_data['degree'].quantile(0.8)
        rich_nodes = tp_data[tp_data['degree'] >= degree_threshold]
        
        # Rich club coefficient (simplified version)
        # In full analysis, this would require connectivity matrices
        rich_club_coef = len(rich_nodes) / len(tp_data)
        
        rich_club_results[timepoint] = {
            'rich_club_coefficient': rich_club_coef,
            'n_rich_nodes': len(rich_nodes),
            'rich_node_ids': list(rich_nodes['node'].values)
        }
    
    return rich_club_results

def core_periphery_analysis(df):
    """
    Analyze core-periphery structure changes.
    Exercise may reorganize core-periphery architecture.
    """
    
    results = {}
    
    for timepoint in df['timepoint'].unique():
        tp_data = df[df['timepoint'] == timepoint]
        
        # Define core nodes (high degree + high betweenness)
        degree_threshold = tp_data['degree'].quantile(0.75)
        betweenness_threshold = tp_data['betweenness'].quantile(0.75)
        
        core_nodes = tp_data[
            (tp_data['degree'] >= degree_threshold) & 
            (tp_data['betweenness'] >= betweenness_threshold)
        ]
        
        # Periphery nodes
        periphery_nodes = tp_data[
            (tp_data['degree'] < degree_threshold) & 
            (tp_data['betweenness'] < betweenness_threshold)
        ]
        
        results[timepoint] = {
            'core_size': len(core_nodes),
            'periphery_size': len(periphery_nodes),
            'core_proportion': len(core_nodes) / len(tp_data),
            'core_nodes': list(core_nodes['node'].values),
            'avg_core_efficiency': core_nodes['local_efficiency'].mean(),
            'avg_periphery_efficiency': periphery_nodes['local_efficiency'].mean()
        }
    
    return results

def network_flexibility_analysis(df):
    """
    Calculate network flexibility - how much nodes change their roles.
    Exercise may increase network flexibility.
    """
    
    # For each node, track its relative ranking across timepoints
    flexibility_scores = {}
    
    nodes = df['node'].unique()
    
    for node in nodes:
        node_data = df[df['node'] == node].sort_values('timepoint')
        
        if len(node_data) == 4:  # Complete data
            # Calculate rank changes for different metrics
            degree_ranks = node_data['degree'].rank(pct=True)
            betweenness_ranks = node_data['betweenness'].rank(pct=True)
            
            # Flexibility = variance in rankings
            degree_flexibility = np.var(degree_ranks)
            betweenness_flexibility = np.var(betweenness_ranks)
            
            flexibility_scores[node] = {
                'degree_flexibility': degree_flexibility,
                'betweenness_flexibility': betweenness_flexibility,
                'combined_flexibility': (degree_flexibility + betweenness_flexibility) / 2
            }
    
    return flexibility_scores

def module_connectivity_analysis(df):
    """
    Analyze within vs between module connectivity changes.
    Exercise may affect modular organization.
    """
    
    # This is a simplified version - full analysis requires community detection
    # on connectivity matrices
    
    # Simulate modules based on node groupings (adjust for your atlas)
    module_assignments = {
        'frontal': list(range(0, 30)),
        'parietal': list(range(30, 60)),
        'temporal': list(range(60, 80)),
        'occipital': list(range(80, 100))
    }
    
    results = {}
    
    for timepoint in df['timepoint'].unique():
        tp_data = df[df['timepoint'] == timepoint]
        
        module_stats = {}
        
        for module_name, nodes in module_assignments.items():
            module_data = tp_data[tp_data['node'].isin(nodes)]
            
            if len(module_data) > 0:
                module_stats[module_name] = {
                    'avg_degree': module_data['degree'].mean(),
                    'avg_clustering': module_data['clustering'].mean(),
                    'avg_efficiency': module_data['local_efficiency'].mean(),
                    'n_nodes': len(module_data)
                }
        
        results[timepoint] = module_stats
    
    return results

def network_resilience_analysis(df):
    """
    Estimate network resilience to node removal.
    Exercise may improve network robustness.
    """
    
    resilience_results = {}
    
    for timepoint in df['timepoint'].unique():
        tp_data = df[df['timepoint'] == timepoint]
        
        # Sort nodes by importance (degree * betweenness)
        tp_data['importance'] = tp_data['degree'] * tp_data['betweenness']
        tp_data_sorted = tp_data.sort_values('importance', ascending=False)
        
        # Simulate targeted attacks (remove most important nodes first)
        n_nodes = len(tp_data)
        removal_proportions = [0.05, 0.10, 0.15, 0.20]  # Remove 5%, 10%, 15%, 20%
        
        attack_resilience = {}
        
        for prop in removal_proportions:
            n_remove = int(n_nodes * prop)
            remaining_nodes = tp_data_sorted.iloc[n_remove:]
            
            # Calculate remaining network efficiency
            remaining_efficiency = remaining_nodes['local_efficiency'].mean()
            efficiency_loss = 1 - (remaining_efficiency / tp_data['local_efficiency'].mean())
            
            attack_resilience[f'remove_{prop:.0%}'] = {
                'efficiency_loss': efficiency_loss,
                'remaining_efficiency': remaining_efficiency,
                'nodes_removed': n_remove
            }
        
        # Simulate random failures
        random_resilience = {}
        
        for prop in removal_proportions:
            n_remove = int(n_nodes * prop)
            
            # Average over 100 random removals
            efficiency_losses = []
            for _ in range(100):
                random_sample = tp_data.sample(n=len(tp_data) - n_remove)
                remaining_efficiency = random_sample['local_efficiency'].mean()
                efficiency_loss = 1 - (remaining_efficiency / tp_data['local_efficiency'].mean())
                efficiency_losses.append(efficiency_loss)
            
            random_resilience[f'remove_{prop:.0%}'] = {
                'mean_efficiency_loss': np.mean(efficiency_losses),
                'std_efficiency_loss': np.std(efficiency_losses)
            }
        
        resilience_results[timepoint] = {
            'targeted_attack': attack_resilience,
            'random_failure': random_resilience
        }
    
    return resilience_results

def temporal_network_analysis(df):
    """
    Analyze temporal dynamics of network changes.
    """
    
    # Calculate session-to-session changes
    temporal_changes = {}
    
    nodes = df['node'].unique()
    
    for node in nodes:
        node_data = df[df['node'] == node].sort_values('timepoint')
        
        if len(node_data) == 4:
            changes = {}
            
            for metric in ['degree', 'betweenness', 'local_efficiency', 'clustering']:
                values = node_data[metric].values
                
                # Calculate changes between consecutive timepoints
                changes[f'{metric}_change_1to2'] = values[1] - values[0]  # Control period
                changes[f'{metric}_change_2to3'] = values[2] - values[1]  # Early training
                changes[f'{metric}_change_3to4'] = values[3] - values[2]  # Late training
                
                # Calculate cumulative changes
                changes[f'{metric}_cumulative_change'] = values[3] - values[0]
                
                # Calculate volatility (sum of absolute changes)
                changes[f'{metric}_volatility'] = np.sum(np.abs(np.diff(values)))
            
            temporal_changes[node] = changes
    
    return temporal_changes

if __name__ == "__main__":
    print("Network-specific analysis functions ready!")
    print("These require nodal metrics data and can provide insights into:")
    print("- Rich club organization")
    print("- Core-periphery structure")  
    print("- Network flexibility")
    print("- Module connectivity")
    print("- Network resilience")
    print("- Temporal dynamics")
