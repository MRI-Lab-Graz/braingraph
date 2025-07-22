#!/usr/bin/env python3
"""
Nodal analysis for running training effects.
Focus on brain regions known to be affected by exercise.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

def analyze_nodal_training_effects(nodal_file="graph_metrics_nodal.csv", output_dir="nodal_analysis_results"):
    """
    Analyze training effects at the nodal level.
    Focus on regions known to be affected by aerobic exercise.
    """
    
    # Load nodal data
    df = pd.read_csv(nodal_file)
    print(f"Loaded nodal data: {df.shape}")
    
    # Exercise-relevant brain regions (adjust node numbers based on your atlas)
    # These are typical regions affected by aerobic exercise
    exercise_regions = {
        'frontal_regions': list(range(0, 20)),      # Prefrontal cortex
        'motor_regions': list(range(20, 35)),       # Motor cortex  
        'hippocampal_regions': list(range(80, 90)), # Hippocampus
        'default_mode': list(range(60, 80)),        # Default mode network
        'executive_control': list(range(35, 50)),   # Executive control
        'visual_regions': list(range(90, 100))      # Visual cortex
    }
    
    # Calculate regional averages
    regional_results = {}
    
    for region_name, nodes in exercise_regions.items():
        print(f"\nAnalyzing {region_name}...")
        
        # Filter data for this region
        region_data = df[df['node'].isin(nodes)].copy()
        
        if len(region_data) == 0:
            print(f"No data found for {region_name}")
            continue
            
        # Average across nodes within region
        region_avg = region_data.groupby(['subject', 'timepoint']).agg({
            'degree': 'mean',
            'strength': 'mean', 
            'eigenvector_centrality': 'mean',
            'betweenness': 'mean',
            'local_efficiency': 'mean',
            'clustering': 'mean'
        }).reset_index()
        
        # Analyze each metric for this region
        region_results = {}
        
        for metric in ['degree', 'strength', 'eigenvector_centrality', 'betweenness', 'local_efficiency', 'clustering']:
            # Calculate period-specific changes (same as global analysis)
            changes = []
            for subject in region_avg['subject'].unique():
                subj_data = region_avg[region_avg['subject'] == subject].sort_values('timepoint')
                
                if len(subj_data) >= 4:
                    control_change = subj_data.iloc[1][metric] - subj_data.iloc[0][metric]
                    training_change = subj_data.iloc[3][metric] - subj_data.iloc[1][metric]
                    
                    changes.append({
                        'subject': subject,
                        'control_change': control_change,
                        'training_change': training_change,
                        'training_vs_control': training_change - control_change
                    })
            
            if len(changes) > 0:
                changes_df = pd.DataFrame(changes)
                
                # Statistical test
                t_stat, p_val = stats.ttest_rel(changes_df['training_change'], changes_df['control_change'])
                effect_size = np.mean(changes_df['training_vs_control']) / np.std(changes_df['training_vs_control'])
                
                region_results[metric] = {
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'mean_effect': np.mean(changes_df['training_vs_control']),
                    'n_subjects': len(changes_df)
                }
        
        regional_results[region_name] = region_results
        
        # Print significant results
        sig_metrics = [m for m, r in region_results.items() if r['p_value'] < 0.05]
        if sig_metrics:
            print(f"  ✅ Significant effects in {region_name}: {sig_metrics}")
        else:
            print(f"  ❌ No significant effects in {region_name}")
    
    return regional_results

def hub_disruption_analysis(nodal_file="graph_metrics_nodal.csv"):
    """
    Analyze changes in network hubs.
    Exercise may reorganize hub structure.
    """
    
    df = pd.read_csv(nodal_file)
    
    results = {}
    
    for timepoint in [1, 2, 3, 4]:
        tp_data = df[df['timepoint'] == timepoint]
        
        # Identify hubs (top 20% by degree or eigenvector centrality)
        degree_threshold = tp_data['degree'].quantile(0.8)
        eig_threshold = tp_data['eigenvector_centrality'].quantile(0.8)
        
        degree_hubs = tp_data[tp_data['degree'] >= degree_threshold]['node'].unique()
        eig_hubs = tp_data[tp_data['eigenvector_centrality'] >= eig_threshold]['node'].unique()
        
        results[f'timepoint_{timepoint}'] = {
            'degree_hubs': list(degree_hubs),
            'eigenvector_hubs': list(eig_hubs),
            'hub_overlap': len(set(degree_hubs) & set(eig_hubs))
        }
    
    # Analyze hub stability across training
    baseline_degree_hubs = set(results['timepoint_1']['degree_hubs'])
    post_training_degree_hubs = set(results['timepoint_4']['degree_hubs'])
    
    hub_stability = len(baseline_degree_hubs & post_training_degree_hubs) / len(baseline_degree_hubs)
    hub_disruption = 1 - hub_stability
    
    print(f"Hub disruption index: {hub_disruption:.3f}")
    print(f"Stable hubs: {len(baseline_degree_hubs & post_training_degree_hubs)}/{len(baseline_degree_hubs)}")
    
    return results, hub_disruption

if __name__ == "__main__":
    # Run nodal analysis
    regional_results = analyze_nodal_training_effects()
    
    # Run hub analysis  
    hub_results, disruption = hub_disruption_analysis()
    
    print(f"\nHub disruption from training: {disruption:.3f}")
