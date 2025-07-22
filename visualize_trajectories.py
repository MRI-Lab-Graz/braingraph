#!/usr/bin/env python3
"""
Visualize training effects across all timepoints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the data for trajectory analysis"""
    # Load the data
    df = pd.read_csv('graph_metrics_global.csv')
    
    # Ensure subject is string for grouping
    df['subject'] = df['subject'].astype(str)
    
    # Filter to complete subjects only
    complete_subjects = df.groupby('subject')['timepoint'].count()
    complete_subjects = complete_subjects[complete_subjects == 4].index.tolist()
    df = df[df['subject'].isin(complete_subjects)]
    
    print(f"✅ Data loaded: {len(complete_subjects)} complete subjects")
    return df, complete_subjects

def create_trajectory_plots():
    """Create comprehensive trajectory visualizations"""
    df, complete_subjects = load_and_prepare_data()
    
    # Network metrics to analyze
    metrics = ['global_efficiency', 'transitivity', 'modularity', 
               'clustering_coef', 'char_path_length']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot individual trajectories (light lines)
        for subject in complete_subjects[:15]:  # Show first 15 for clarity
            subj_data = df[df['subject'] == subject].sort_values('timepoint')
            ax.plot(subj_data['timepoint'], subj_data[metric], 
                   'o-', alpha=0.3, color='gray', linewidth=1, markersize=3)
        
        # Calculate and plot mean trajectory with confidence intervals
        mean_trajectory = df.groupby('timepoint')[metric].mean()
        sem_trajectory = df.groupby('timepoint')[metric].sem()
        
        ax.plot(mean_trajectory.index, mean_trajectory.values, 
               'o-', color='red', linewidth=4, markersize=8, 
               label='Mean ± SEM', zorder=10)
        
        ax.fill_between(mean_trajectory.index, 
                       mean_trajectory.values - sem_trajectory,
                       mean_trajectory.values + sem_trajectory,
                       alpha=0.3, color='red', zorder=5)
        
        # Add training period shading
        ax.axvspan(2, 4, alpha=0.1, color='blue', label='Training Period')
        ax.axvline(x=2, color='blue', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Training Start')
        
        # Statistical annotations
        # Compare T1 vs T2 (control period)
        t1_data = df[df['timepoint'] == 1][metric]
        t2_data = df[df['timepoint'] == 2][metric]
        _, p_control = stats.ttest_rel(t2_data, t1_data)
        
        # Compare T2 vs T4 (training period)
        t4_data = df[df['timepoint'] == 4][metric]
        _, p_training = stats.ttest_rel(t4_data, t2_data)
        
        # Training effect (contrast analysis)
        control_change = t2_data - t1_data
        training_change = t4_data - t2_data
        training_effect = training_change - control_change
        _, p_effect = stats.ttest_1samp(training_effect, 0)
        
        # Add statistical text
        ax.text(0.02, 0.98, f'Control: p={p_control:.3f}\nTraining: p={p_training:.3f}\nEffect: p={p_effect:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
        
        # Formatting
        ax.set_xlabel('Session', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()}\nLongitudinal Trajectory', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['T1\n(Baseline)', 'T2\n(Pre-training)', 
                           'T3\n(Mid-training)', 'T4\n(Post-training)'])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10)
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/Trajectory_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_change_score_analysis():
    """Create detailed change score analysis"""
    df, complete_subjects = load_and_prepare_data()
    
    metrics = ['global_efficiency', 'transitivity', 'modularity', 
               'clustering_coef', 'char_path_length']
    
    # Calculate change scores for each participant
    change_data = []
    
    for subject in complete_subjects:
        subj_data = df[df['subject'] == subject].sort_values('timepoint')
        if len(subj_data) == 4:
            t1, t2, t3, t4 = subj_data.iloc[0], subj_data.iloc[1], subj_data.iloc[2], subj_data.iloc[3]
            
            for metric in metrics:
                control_change = t2[metric] - t1[metric]
                training_change = t4[metric] - t2[metric]
                net_effect = training_change - control_change
                
                change_data.append({
                    'subject': subject,
                    'metric': metric,
                    'control_change': control_change,
                    'training_change': training_change,
                    'net_effect': net_effect,
                    'baseline_value': t1[metric]
                })
    
    change_df = pd.DataFrame(change_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Change scores comparison
    ax1 = axes[0, 0]
    significant_metrics = ['global_efficiency', 'transitivity']
    
    for i, metric in enumerate(significant_metrics):
        metric_data = change_df[change_df['metric'] == metric]
        
        x_pos = np.array([i*3, i*3+1])
        means = [metric_data['control_change'].mean(), metric_data['training_change'].mean()]
        sems = [metric_data['control_change'].sem(), metric_data['training_change'].sem()]
        
        bars = ax1.bar(x_pos, means, yerr=sems, capsize=5, 
                      color=['lightblue', 'orange'], alpha=0.7,
                      label=['Control', 'Training'] if i == 0 else "")
        
        # Add individual points
        ax1.scatter([i*3]*len(metric_data), metric_data['control_change'], 
                   alpha=0.6, color='blue', s=30)
        ax1.scatter([i*3+1]*len(metric_data), metric_data['training_change'], 
                   alpha=0.6, color='red', s=30)
        
        # Connect paired observations
        for _, row in metric_data.iterrows():
            ax1.plot([i*3, i*3+1], [row['control_change'], row['training_change']], 
                    'k-', alpha=0.3, linewidth=0.5)
    
    ax1.set_title('Change Scores: Control vs Training Periods', fontweight='bold')
    ax1.set_ylabel('Change Score', fontweight='bold')
    ax1.set_xticks([0.5, 3.5])
    ax1.set_xticklabels(['Global Efficiency', 'Transitivity'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Net training effects
    ax2 = axes[0, 1]
    
    for i, metric in enumerate(metrics):
        metric_data = change_df[change_df['metric'] == metric]
        net_effects = metric_data['net_effect']
        
        # Box plot
        bp = ax2.boxplot([net_effects], positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        
        # Add individual points
        ax2.scatter([i]*len(net_effects), net_effects, alpha=0.6, color='darkgreen', s=30)
        
        # Test against zero
        _, p_val = stats.ttest_1samp(net_effects, 0)
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        
        ax2.text(i, max(net_effects) + 0.001, significance, 
                ha='center', va='bottom', fontweight='bold')
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Net Training Effects', fontweight='bold')
    ax2.set_ylabel('Net Effect (Training - Control)', fontweight='bold')
    ax2.set_xlabel('Network Metric', fontweight='bold')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m.replace('_', '\n').title() for m in metrics], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation with baseline
    ax3 = axes[1, 0]
    
    for metric in significant_metrics:
        metric_data = change_df[change_df['metric'] == metric]
        
        ax3.scatter(metric_data['baseline_value'], metric_data['net_effect'], 
                   alpha=0.7, s=50, label=metric.replace('_', ' ').title())
        
        # Add correlation line
        r, p = stats.pearsonr(metric_data['baseline_value'], metric_data['net_effect'])
        if p < 0.05:
            z = np.polyfit(metric_data['baseline_value'], metric_data['net_effect'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(metric_data['baseline_value'].min(), 
                               metric_data['baseline_value'].max(), 100)
            ax3.plot(x_line, p_line(x_line), '--', alpha=0.8)
            ax3.text(0.05, 0.95 - 0.1*list(significant_metrics).index(metric), 
                    f'{metric}: r={r:.3f}, p={p:.3f}',
                    transform=ax3.transAxes, fontsize=10)
    
    ax3.set_title('Training Response vs Baseline', fontweight='bold')
    ax3.set_xlabel('Baseline Value', fontweight='bold')
    ax3.set_ylabel('Net Training Effect', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Effect size summary
    ax4 = axes[1, 1]
    
    effect_sizes = []
    p_values = []
    
    for metric in metrics:
        metric_data = change_df[change_df['metric'] == metric]
        net_effects = metric_data['net_effect']
        
        # Cohen's d
        d = net_effects.mean() / net_effects.std()
        effect_sizes.append(d)
        
        # p-value
        _, p = stats.ttest_1samp(net_effects, 0)
        p_values.append(p)
    
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars = ax4.barh(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
    
    # Add effect size thresholds
    ax4.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    ax4.set_title('Effect Sizes (Cohen\'s d)', fontweight='bold')
    ax4.set_xlabel('Effect Size', fontweight='bold')
    ax4.set_yticks(range(len(metrics)))
    ax4.set_yticklabels([m.replace('_', '\n').title() for m in metrics])
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/Change_Score_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return change_df

def statistical_summary_table():
    """Create comprehensive statistical summary"""
    df, complete_subjects = load_and_prepare_data()
    
    metrics = ['global_efficiency', 'transitivity', 'modularity', 
               'clustering_coef', 'char_path_length']
    
    results = []
    
    for metric in metrics:
        # Get data for each timepoint
        t1 = df[df['timepoint'] == 1][metric].values
        t2 = df[df['timepoint'] == 2][metric].values
        t3 = df[df['timepoint'] == 3][metric].values
        t4 = df[df['timepoint'] == 4][metric].values
        
        # Calculate changes
        control_change = t2 - t1
        training_change = t4 - t2
        net_effect = training_change - control_change
        
        # Statistics
        _, p_control = stats.ttest_1samp(control_change, 0)
        _, p_training = stats.ttest_1samp(training_change, 0)
        _, p_effect = stats.ttest_1samp(net_effect, 0)
        
        # Effect sizes
        d_control = control_change.mean() / control_change.std() if control_change.std() > 0 else 0
        d_training = training_change.mean() / training_change.std() if training_change.std() > 0 else 0
        d_effect = net_effect.mean() / net_effect.std() if net_effect.std() > 0 else 0
        
        results.append({
            'Metric': metric.replace('_', ' ').title(),
            'Control_Change_Mean': f"{control_change.mean():.4f}",
            'Control_Change_SEM': f"{stats.sem(control_change):.4f}",
            'Control_p': f"{p_control:.3f}",
            'Control_d': f"{d_control:.3f}",
            'Training_Change_Mean': f"{training_change.mean():.4f}",
            'Training_Change_SEM': f"{stats.sem(training_change):.4f}",
            'Training_p': f"{p_training:.3f}",
            'Training_d': f"{d_training:.3f}",
            'Net_Effect_Mean': f"{net_effect.mean():.4f}",
            'Net_Effect_SEM': f"{stats.sem(net_effect):.4f}",
            'Net_Effect_p': f"{p_effect:.3f}",
            'Net_Effect_d': f"{d_effect:.3f}",
            'Significant': 'Yes' if p_effect < 0.05 else 'No'
        })
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv('comprehensive_results/trajectory_statistical_summary.csv', index=False)
    
    print("📊 COMPREHENSIVE TRAJECTORY ANALYSIS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    print("🧠 BRAIN NETWORK TRAJECTORY ANALYSIS")
    print("="*50)
    
    # Create output directory
    import os
    os.makedirs('comprehensive_results', exist_ok=True)
    
    # Run analyses
    print("\n1️⃣ Creating trajectory plots...")
    df = create_trajectory_plots()
    
    print("\n2️⃣ Creating change score analysis...")
    change_df = create_change_score_analysis()
    
    print("\n3️⃣ Generating statistical summary...")
    summary_df = statistical_summary_table()
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: comprehensive_results/")
    print(f"   📊 Trajectory_Analysis.png")
    print(f"   📊 Change_Score_Analysis.png") 
    print(f"   📄 trajectory_statistical_summary.csv")
