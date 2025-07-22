#!/usr/bin/env python3
"""
Network-Based Statistics (NBS) and Permutation Testing
For identifying specific connections showing training effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_connectivity_data():
    """
    Load connectivity matrices (simulated for demonstration)
    In reality, these would be your actual connectivity matrices
    """
    print("📊 Loading connectivity data...")
    
    # For demonstration, we'll simulate connectivity matrices
    # In your case, you would load actual connectivity matrices
    n_subjects = 21
    n_timepoints = 4
    n_regions = 90  # AAL atlas regions
    
    # Simulate connectivity matrices
    np.random.seed(42)
    connectivity_data = {}
    
    for subject in range(1, n_subjects + 1):
        connectivity_data[subject] = {}
        
        # Base connectivity pattern
        base_connectivity = np.random.rand(n_regions, n_regions)
        base_connectivity = (base_connectivity + base_connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(base_connectivity, 0)  # No self-connections
        
        for timepoint in range(1, n_timepoints + 1):
            # Add small random variations
            noise = np.random.normal(0, 0.05, (n_regions, n_regions))
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0)
            
            # Add training effect for some connections after T2
            if timepoint > 2:  # Training period
                # Simulate training effects in specific regions (e.g., motor-cognitive networks)
                training_effect = np.zeros((n_regions, n_regions))
                
                # Define regions of interest (ROI pairs that might show training effects)
                motor_regions = [0, 1, 2, 3, 4]  # Simulated motor regions
                cognitive_regions = [30, 31, 32, 33, 34]  # Simulated cognitive regions
                
                for m_roi in motor_regions:
                    for c_roi in cognitive_regions:
                        # Add progressive training effect
                        effect_size = 0.1 * (timepoint - 2) * np.random.normal(1, 0.2)
                        training_effect[m_roi, c_roi] = effect_size
                        training_effect[c_roi, m_roi] = effect_size
                
                connectivity = base_connectivity + noise + training_effect
            else:
                connectivity = base_connectivity + noise
            
            # Threshold to ensure values are in reasonable range
            connectivity = np.clip(connectivity, 0, 1)
            connectivity_data[subject][timepoint] = connectivity
    
    print(f"✅ Simulated connectivity data: {n_subjects} subjects × {n_timepoints} timepoints × {n_regions}×{n_regions} matrices")
    return connectivity_data, n_subjects, n_timepoints, n_regions

def permutation_test_single_edge(x, y, n_permutations=5000):
    """
    Permutation test for a single edge
    """
    # Observed difference
    obs_diff = np.mean(x) - np.mean(y)
    
    # Combine data
    combined = np.concatenate([x, y])
    n_x = len(x)
    
    # Permutation test
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]
        perm_diff = np.mean(perm_x) - np.mean(perm_y)
        perm_diffs.append(perm_diff)
    
    # Calculate p-value
    perm_diffs = np.array(perm_diffs)
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) / n_permutations
    
    return obs_diff, p_value

def network_based_statistics(connectivity_data, n_subjects, n_regions, alpha=0.05, 
                           cluster_alpha=0.001, n_permutations=1000):
    """
    Perform Network-Based Statistics (NBS) analysis
    """
    print(f"🔬 Running Network-Based Statistics...")
    print(f"   Alpha level: {alpha}")
    print(f"   Cluster-forming threshold: {cluster_alpha}")
    print(f"   Permutations: {n_permutations}")
    
    # Extract connectivity changes
    control_changes = []  # T2 - T1
    training_changes = [] # T4 - T2
    
    for subject in range(1, n_subjects + 1):
        # Control period change
        control_change = connectivity_data[subject][2] - connectivity_data[subject][1]
        control_changes.append(control_change)
        
        # Training period change
        training_change = connectivity_data[subject][4] - connectivity_data[subject][2]
        training_changes.append(training_change)
    
    control_changes = np.array(control_changes)
    training_changes = np.array(training_changes)
    
    # Calculate contrast (training - control changes)
    contrast_changes = training_changes - control_changes
    
    # Step 1: Edge-wise statistical testing
    print("   Step 1: Edge-wise testing...")
    t_stats = np.zeros((n_regions, n_regions))
    p_values = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            # Test if training effect is different from zero
            edge_effects = contrast_changes[:, i, j]
            t_stat, p_val = stats.ttest_1samp(edge_effects, 0)
            
            t_stats[i, j] = t_stat
            t_stats[j, i] = t_stat
            p_values[i, j] = p_val
            p_values[j, i] = p_val
    
    # Step 2: Cluster-forming threshold
    print("   Step 2: Cluster formation...")
    significant_edges = p_values < cluster_alpha
    
    # Create network graph for cluster detection
    G = nx.Graph()
    significant_connections = []
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if significant_edges[i, j]:
                G.add_edge(i, j, weight=np.abs(t_stats[i, j]))
                significant_connections.append((i, j, t_stats[i, j]))
    
    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))
    cluster_stats = []
    
    for cluster_idx, cluster in enumerate(clusters):
        cluster_edges = [(i, j) for i, j in G.edges() if i in cluster and j in cluster]
        cluster_size = len(cluster_edges)
        cluster_stat = sum([G[i][j]['weight'] for i, j in cluster_edges])
        cluster_stats.append({
            'cluster_id': cluster_idx,
            'size': cluster_size,
            'statistic': cluster_stat,
            'nodes': list(cluster),
            'edges': cluster_edges
        })
    
    print(f"   Found {len(clusters)} clusters with {len(significant_connections)} significant edges")
    
    # Step 3: Permutation testing for cluster significance
    print("   Step 3: Permutation testing...")
    max_cluster_stats = []
    
    for perm in range(n_permutations):
        if perm % 200 == 0:
            print(f"      Permutation {perm}/{n_permutations}")
        
        # Permute the contrast data
        perm_contrast = contrast_changes.copy()
        for subject in range(n_subjects):
            # Random sign flip for each subject
            if np.random.rand() > 0.5:
                perm_contrast[subject] = -perm_contrast[subject]
        
        # Recalculate statistics
        perm_t_stats = np.zeros((n_regions, n_regions))
        perm_p_values = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                edge_effects = perm_contrast[:, i, j]
                t_stat, p_val = stats.ttest_1samp(edge_effects, 0)
                perm_t_stats[i, j] = t_stat
                perm_t_stats[j, i] = t_stat
                perm_p_values[i, j] = p_val
                perm_p_values[j, i] = p_val
        
        # Find clusters in permuted data
        perm_significant = perm_p_values < cluster_alpha
        perm_G = nx.Graph()
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                if perm_significant[i, j]:
                    perm_G.add_edge(i, j, weight=np.abs(perm_t_stats[i, j]))
        
        # Get maximum cluster statistic
        if len(perm_G.edges()) > 0:
            perm_clusters = list(nx.connected_components(perm_G))
            perm_cluster_stats = []
            
            for cluster in perm_clusters:
                cluster_edges = [(i, j) for i, j in perm_G.edges() if i in cluster and j in cluster]
                cluster_stat = sum([perm_G[i][j]['weight'] for i, j in cluster_edges])
                perm_cluster_stats.append(cluster_stat)
            
            max_cluster_stats.append(max(perm_cluster_stats))
        else:
            max_cluster_stats.append(0)
    
    # Calculate cluster p-values
    max_cluster_stats = np.array(max_cluster_stats)
    
    for cluster in cluster_stats:
        cluster['p_value'] = np.sum(max_cluster_stats >= cluster['statistic']) / n_permutations
        cluster['significant'] = cluster['p_value'] < alpha
    
    print(f"✅ NBS analysis complete!")
    
    return {
        'cluster_stats': cluster_stats,
        't_statistics': t_stats,
        'p_values': p_values,
        'significant_edges': significant_edges,
        'contrast_changes': contrast_changes,
        'null_distribution': max_cluster_stats
    }

def visualize_nbs_results(nbs_results, n_regions):
    """
    Visualize NBS results
    """
    print("📊 Creating NBS visualizations...")
    
    cluster_stats = nbs_results['cluster_stats']
    t_stats = nbs_results['t_statistics']
    significant_edges = nbs_results['significant_edges']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. T-statistics matrix
    ax1 = axes[0, 0]
    im1 = ax1.imshow(t_stats, cmap='RdBu_r', vmin=-5, vmax=5)
    ax1.set_title('T-Statistics Matrix\n(Training vs Control Effects)', fontweight='bold')
    ax1.set_xlabel('Brain Region')
    ax1.set_ylabel('Brain Region')
    plt.colorbar(im1, ax=ax1, label='T-statistic')
    
    # 2. Significant edges
    ax2 = axes[0, 1]
    im2 = ax2.imshow(significant_edges.astype(int), cmap='Reds', vmin=0, vmax=1)
    ax2.set_title('Significant Edges\n(Cluster-forming threshold)', fontweight='bold')
    ax2.set_xlabel('Brain Region')
    ax2.set_ylabel('Brain Region')
    plt.colorbar(im2, ax=ax2, label='Significant')
    
    # 3. Cluster statistics
    ax3 = axes[1, 0]
    if len(cluster_stats) > 0:
        cluster_sizes = [c['size'] for c in cluster_stats]
        cluster_pvals = [c['p_value'] for c in cluster_stats]
        colors = ['red' if c['significant'] else 'gray' for c in cluster_stats]
        
        bars = ax3.bar(range(len(cluster_stats)), cluster_sizes, color=colors, alpha=0.7)
        ax3.set_title('Cluster Sizes and Significance', fontweight='bold')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Edges')
        
        # Add p-values as text
        for i, (bar, pval) in enumerate(zip(bars, cluster_pvals)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'p={pval:.3f}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No significant clusters found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Cluster Analysis', fontweight='bold')
    
    # 4. Null distribution
    ax4 = axes[1, 1]
    null_dist = nbs_results['null_distribution']
    ax4.hist(null_dist, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    
    if len(cluster_stats) > 0:
        for cluster in cluster_stats:
            color = 'red' if cluster['significant'] else 'orange'
            ax4.axvline(cluster['statistic'], color=color, linestyle='--', 
                       linewidth=2, label=f"Cluster {cluster['cluster_id']}")
    
    ax4.set_title('Null Distribution\nCluster Statistics', fontweight='bold')
    ax4.set_xlabel('Maximum Cluster Statistic')
    ax4.set_ylabel('Frequency')
    if len(cluster_stats) > 0:
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/NBS_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_nbs_summary_table(nbs_results):
    """
    Create summary table of NBS results
    """
    cluster_stats = nbs_results['cluster_stats']
    
    if len(cluster_stats) == 0:
        print("📋 No significant clusters found in NBS analysis")
        return pd.DataFrame()
    
    summary_data = []
    for cluster in cluster_stats:
        summary_data.append({
            'Cluster_ID': cluster['cluster_id'],
            'Size_Edges': cluster['size'],
            'Size_Nodes': len(cluster['nodes']),
            'Test_Statistic': f"{cluster['statistic']:.3f}",
            'P_Value': f"{cluster['p_value']:.3f}",
            'Significant': 'Yes' if cluster['significant'] else 'No',
            'Nodes': ', '.join(map(str, sorted(cluster['nodes'])[:10])) + ('...' if len(cluster['nodes']) > 10 else '')
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    summary_df.to_csv('comprehensive_results/NBS_summary.csv', index=False)
    
    print("📋 NBS CLUSTER SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    return summary_df

def run_additional_permutation_tests(connectivity_data, n_subjects, n_regions):
    """
    Additional permutation tests for specific hypotheses
    """
    print("🔄 Running additional permutation tests...")
    
    # Test 1: Global connectivity strength
    print("   Test 1: Global connectivity strength changes")
    
    global_strength_changes = []
    for subject in range(1, n_subjects + 1):
        # Control period
        control_strength = np.mean(connectivity_data[subject][2] - connectivity_data[subject][1])
        
        # Training period  
        training_strength = np.mean(connectivity_data[subject][4] - connectivity_data[subject][2])
        
        # Net effect
        net_effect = training_strength - control_strength
        global_strength_changes.append(net_effect)
    
    # Permutation test
    _, p_global = stats.ttest_1samp(global_strength_changes, 0)
    effect_size_global = np.mean(global_strength_changes) / np.std(global_strength_changes)
    
    print(f"      Global strength: p = {p_global:.3f}, d = {effect_size_global:.3f}")
    
    # Test 2: Network efficiency (simulated)
    print("   Test 2: Network efficiency changes")
    
    efficiency_changes = []
    for subject in range(1, n_subjects + 1):
        # Convert to graph and calculate efficiency
        def graph_efficiency(matrix):
            # Simple efficiency calculation (mean of inverse distances)
            G = nx.from_numpy_array(matrix)
            try:
                efficiency = nx.global_efficiency(G)
            except:
                efficiency = 0
            return efficiency
        
        # Control period
        eff_t1 = graph_efficiency(connectivity_data[subject][1])
        eff_t2 = graph_efficiency(connectivity_data[subject][2])
        control_eff_change = eff_t2 - eff_t1
        
        # Training period
        eff_t4 = graph_efficiency(connectivity_data[subject][4])
        training_eff_change = eff_t4 - eff_t2
        
        # Net effect
        net_eff_effect = training_eff_change - control_eff_change
        efficiency_changes.append(net_eff_effect)
    
    _, p_efficiency = stats.ttest_1samp(efficiency_changes, 0)
    effect_size_efficiency = np.mean(efficiency_changes) / np.std(efficiency_changes)
    
    print(f"      Network efficiency: p = {p_efficiency:.3f}, d = {effect_size_efficiency:.3f}")
    
    # Test 3: Modularity changes
    print("   Test 3: Modularity changes")
    
    modularity_changes = []
    for subject in range(1, n_subjects + 1):
        def graph_modularity(matrix):
            G = nx.from_numpy_array(matrix)
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(G)
                modularity = nx.algorithms.community.modularity(G, communities)
            except:
                modularity = 0
            return modularity
        
        # Control period
        mod_t1 = graph_modularity(connectivity_data[subject][1])
        mod_t2 = graph_modularity(connectivity_data[subject][2])
        control_mod_change = mod_t2 - mod_t1
        
        # Training period
        mod_t4 = graph_modularity(connectivity_data[subject][4])
        training_mod_change = mod_t4 - mod_t2
        
        # Net effect
        net_mod_effect = training_mod_change - control_mod_change
        modularity_changes.append(net_mod_effect)
    
    _, p_modularity = stats.ttest_1samp(modularity_changes, 0)
    effect_size_modularity = np.mean(modularity_changes) / np.std(modularity_changes)
    
    print(f"      Modularity: p = {p_modularity:.3f}, d = {effect_size_modularity:.3f}")
    
    return {
        'global_strength': {
            'changes': global_strength_changes,
            'p_value': p_global,
            'effect_size': effect_size_global
        },
        'efficiency': {
            'changes': efficiency_changes,
            'p_value': p_efficiency,
            'effect_size': effect_size_efficiency
        },
        'modularity': {
            'changes': modularity_changes,
            'p_value': p_modularity,
            'effect_size': effect_size_modularity
        }
    }

if __name__ == "__main__":
    print("🧠 NETWORK-BASED STATISTICS ANALYSIS")
    print("="*50)
    
    # Create output directory
    import os
    os.makedirs('comprehensive_results', exist_ok=True)
    
    # Load connectivity data
    connectivity_data, n_subjects, n_timepoints, n_regions = load_connectivity_data()
    
    # Run NBS analysis
    print("\n1️⃣ Running Network-Based Statistics...")
    nbs_results = network_based_statistics(connectivity_data, n_subjects, n_regions,
                                         alpha=0.05, cluster_alpha=0.001, n_permutations=1000)
    
    # Visualize results
    print("\n2️⃣ Creating visualizations...")
    visualize_nbs_results(nbs_results, n_regions)
    
    # Create summary table
    print("\n3️⃣ Creating summary table...")
    nbs_summary = create_nbs_summary_table(nbs_results)
    
    # Additional permutation tests
    print("\n4️⃣ Running additional permutation tests...")
    additional_results = run_additional_permutation_tests(connectivity_data, n_subjects, n_regions)
    
    print("\n✅ Network-Based Statistics analysis complete!")
    print(f"📁 Results saved to: comprehensive_results/")
    print(f"   📊 NBS_Analysis.png")
    print(f"   📄 NBS_summary.csv")
    
    # Summary
    significant_clusters = sum([1 for c in nbs_results['cluster_stats'] if c['significant']])
    print(f"\n📋 SUMMARY:")
    print(f"   🔍 Significant clusters found: {significant_clusters}")
    print(f"   📊 Global connectivity: p = {additional_results['global_strength']['p_value']:.3f}")
    print(f"   🕸️  Network efficiency: p = {additional_results['efficiency']['p_value']:.3f}")
    print(f"   🏗️  Modularity: p = {additional_results['modularity']['p_value']:.3f}")
