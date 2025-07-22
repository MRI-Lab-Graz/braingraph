#!/usr/bin/env python3
"""
Create publication-quality figures for the running training study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def create_figure_1_study_design():
    """Figure 1: Study design and timeline"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Panel A: Experimental timeline
    sessions = [1, 2, 3, 4]
    session_labels = ['Baseline\n(T1)', 'Pre-training\n(T2)', 'Mid-training\n(T3)', 'Post-training\n(T4)']
    
    # Timeline
    ax1.plot(sessions, [1]*4, 'ko-', linewidth=3, markersize=10)
    
    # Add period annotations
    ax1.axvspan(1, 2, alpha=0.3, color='lightblue', label='Control Period')
    ax1.axvspan(2, 4, alpha=0.3, color='orange', label='Training Period')
    
    # Add arrows and labels
    ax1.annotate('No Training', xy=(1.5, 1.1), ha='center', fontsize=12, fontweight='bold')
    ax1.annotate('Running Training', xy=(3, 1.1), ha='center', fontsize=12, fontweight='bold')
    
    for i, (s, label) in enumerate(zip(sessions, session_labels)):
        ax1.text(s, 0.9, label, ha='center', va='top', fontsize=11)
    
    ax1.set_xlim(0.5, 4.5)
    ax1.set_ylim(0.8, 1.3)
    ax1.set_xlabel('Session', fontsize=14, fontweight='bold')
    ax1.set_title('A. Experimental Design', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(False)
    ax1.set_yticks([])
    
    # Panel B: Sample characteristics
    sample_data = {
        'Characteristic': ['Sample Size', 'Age (years)', 'Sex (F/M)', 'Training Duration', 'Sessions per Week'],
        'Value': ['N = 22', 'Mean ± SD', '12/10', '8 weeks', '3 sessions']
    }
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=[[char, val] for char, val in zip(sample_data['Characteristic'], sample_data['Value'])],
                     colLabels=['Characteristic', 'Value'],
                     cellLoc='left',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax2.set_title('B. Sample Characteristics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/Figure1_StudyDesign.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_2_main_results():
    """Figure 2: Main training effects"""
    
    # Load results data
    ge_changes = pd.read_csv('comprehensive_results/global_efficiency_changes.csv')
    trans_changes = pd.read_csv('comprehensive_results/transitivity_changes.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Global Efficiency Results
    # Panel A: Bar plot of period comparison
    periods = ['Control\n(T2-T1)', 'Training\n(T4-T2)']
    ge_means = [ge_changes['control_change'].mean(), ge_changes['training_change'].mean()]
    ge_sems = [stats.sem(ge_changes['control_change']), stats.sem(ge_changes['training_change'])]
    
    bars1 = axes[0,0].bar(periods, ge_means, yerr=ge_sems, capsize=5, 
                         color=['lightblue', 'orange'], alpha=0.8, edgecolor='black')
    axes[0,0].set_ylabel('Change in Global Efficiency', fontsize=12, fontweight='bold')
    axes[0,0].set_title('A. Global Efficiency\nPeriod Comparison', fontsize=14, fontweight='bold')
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance annotation
    max_val = max(ge_means) + max(ge_sems)
    axes[0,0].plot([0, 1], [max_val*1.1, max_val*1.1], 'k-', linewidth=1)
    axes[0,0].text(0.5, max_val*1.15, 'p = 0.044*', ha='center', fontsize=12, fontweight='bold')
    
    # Panel B: Effect size distribution
    axes[0,1].hist(ge_changes['training_vs_control'], bins=8, alpha=0.7, color='green', 
                   edgecolor='black', density=True)
    axes[0,1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
    axes[0,1].axvline(x=ge_changes['training_vs_control'].mean(), color='blue', 
                      linewidth=3, label=f'Mean = {ge_changes["training_vs_control"].mean():.4f}')
    axes[0,1].set_xlabel('Training - Control Change', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0,1].set_title('B. Effect Size Distribution\n(Cohen\'s d = 0.48)', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    
    # Panel C: Individual differences scatter
    axes[0,2].scatter(ge_changes['control_change'], ge_changes['training_change'], 
                     s=80, alpha=0.7, color='purple', edgecolor='black')
    
    # Add unity line
    min_val = min(ge_changes['control_change'].min(), ge_changes['training_change'].min())
    max_val = max(ge_changes['control_change'].max(), ge_changes['training_change'].max())
    axes[0,2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No difference')
    
    axes[0,2].set_xlabel('Control Period Change', fontsize=12, fontweight='bold')
    axes[0,2].set_ylabel('Training Period Change', fontsize=12, fontweight='bold')
    axes[0,2].set_title('C. Individual Responses', fontsize=14, fontweight='bold')
    axes[0,2].legend()
    
    # Transitivity Results (same structure)
    # Panel D: Bar plot
    trans_means = [trans_changes['control_change'].mean(), trans_changes['training_change'].mean()]
    trans_sems = [stats.sem(trans_changes['control_change']), stats.sem(trans_changes['training_change'])]
    
    bars2 = axes[1,0].bar(periods, trans_means, yerr=trans_sems, capsize=5,
                         color=['lightblue', 'orange'], alpha=0.8, edgecolor='black')
    axes[1,0].set_ylabel('Change in Transitivity', fontsize=12, fontweight='bold')
    axes[1,0].set_title('D. Transitivity\nPeriod Comparison', fontsize=14, fontweight='bold')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance annotation
    max_val = max(trans_means) + max(trans_sems)
    axes[1,0].plot([0, 1], [max_val*1.1, max_val*1.1], 'k-', linewidth=1)
    axes[1,0].text(0.5, max_val*1.15, 'p = 0.031*', ha='center', fontsize=12, fontweight='bold')
    
    # Panel E: Effect size distribution
    axes[1,1].hist(trans_changes['training_vs_control'], bins=8, alpha=0.7, color='green',
                   edgecolor='black', density=True)
    axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
    axes[1,1].axvline(x=trans_changes['training_vs_control'].mean(), color='blue',
                      linewidth=3, label=f'Mean = {trans_changes["training_vs_control"].mean():.4f}')
    axes[1,1].set_xlabel('Training - Control Change', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1,1].set_title('E. Effect Size Distribution\n(Cohen\'s d = 0.52)', fontsize=14, fontweight='bold')
    axes[1,1].legend()
    
    # Panel F: Individual differences scatter
    axes[1,2].scatter(trans_changes['control_change'], trans_changes['training_change'],
                     s=80, alpha=0.7, color='purple', edgecolor='black')
    
    # Add unity line
    min_val = min(trans_changes['control_change'].min(), trans_changes['training_change'].min())
    max_val = max(trans_changes['control_change'].max(), trans_changes['training_change'].max())
    axes[1,2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No difference')
    
    axes[1,2].set_xlabel('Control Period Change', fontsize=12, fontweight='bold')
    axes[1,2].set_ylabel('Training Period Change', fontsize=12, fontweight='bold')
    axes[1,2].set_title('F. Individual Responses', fontsize=14, fontweight='bold')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/Figure2_MainResults.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_3_comprehensive_results():
    """Figure 3: Comprehensive results across all metrics"""
    
    # Results data (from analysis)
    metrics = ['Global\nEfficiency', 'Transitivity', 'Modularity', 'Clustering\nCoefficient', 'Path\nLength']
    effect_sizes = [0.480, 0.519, -0.071, 0.173, -0.079]
    p_values = [0.0441, 0.0310, 0.7543, 0.4473, 0.7285]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Effect sizes
    colors = ['red' if p < 0.05 else 'lightgray' for p in p_values]
    bars = ax1.barh(range(len(metrics)), effect_sizes, color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics, fontsize=12)
    ax1.set_xlabel("Effect Size (Cohen's d)", fontsize=14, fontweight='bold')
    ax1.set_title("A. Training Effect Sizes", fontsize=16, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium (0.5)')
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9, label='Large (0.8)')
    ax1.legend(title='Effect Size Thresholds', fontsize=10)
    
    # Add effect size values
    for i, (effect, p) in enumerate(zip(effect_sizes, p_values)):
        sig_marker = '*' if p < 0.05 else ''
        ax1.text(effect + 0.05 if effect >= 0 else effect - 0.05, i, 
                f'{effect:.3f}{sig_marker}', va='center', fontweight='bold')
    
    # Panel B: Statistical significance
    neg_log_p = [-np.log10(p) for p in p_values]
    bars2 = ax2.barh(range(len(metrics)), neg_log_p, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels(metrics, fontsize=12)
    ax2.set_xlabel("-log₁₀(p-value)", fontsize=14, fontweight='bold')
    ax2.set_title("B. Statistical Significance", fontsize=16, fontweight='bold')
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axvline(x=-np.log10(0.01), color='red', linestyle='--', alpha=0.9, label='p = 0.01')
    ax2.legend(title='Significance Thresholds', fontsize=10)
    
    # Add p-values
    for i, p in enumerate(p_values):
        ax2.text(-np.log10(p) + 0.05, i, f'p = {p:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/Figure3_ComprehensiveResults.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_supplementary_trajectories():
    """Supplementary Figure: Individual trajectories"""
    
    # Load the enhanced data
    df = pd.read_csv('graph_metrics_global_enhanced.csv')
    
    significant_metrics = ['global_efficiency', 'transitivity']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, metric in enumerate(significant_metrics):
        # Plot individual trajectories
        for subject in df['subject'].unique():
            if subject != '15':  # Exclude incomplete subject
                subj_data = df[df['subject'] == subject].sort_values('timepoint')
                axes[idx].plot(subj_data['timepoint'], subj_data[metric], 
                              'o-', alpha=0.3, color='gray', linewidth=1, markersize=4)
        
        # Plot mean trajectory
        mean_trajectory = df.groupby('timepoint')[metric].mean()
        std_trajectory = df.groupby('timepoint')[metric].std()
        
        axes[idx].plot(mean_trajectory.index, mean_trajectory.values, 
                      'o-', color='red', linewidth=3, markersize=8, label='Mean ± SEM')
        sem_values = df.groupby('timepoint')[metric].sem()
        axes[idx].fill_between(mean_trajectory.index, 
                              mean_trajectory.values - sem_values,
                              mean_trajectory.values + sem_values,
                              alpha=0.3, color='red')
        
        # Add training start line
        axes[idx].axvline(x=2, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Training Start')
        
        # Formatting
        axes[idx].set_xlabel('Session', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}\nTrajectories', 
                           fontsize=16, fontweight='bold')
        axes[idx].set_xticks([1, 2, 3, 4])
        axes[idx].set_xticklabels(['Baseline', 'Pre-training', 'Mid-training', 'Post-training'])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/SupplementaryFigure_Trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating publication-quality figures...")
    
    # Create figures
    create_figure_1_study_design()
    print("✅ Figure 1: Study design created")
    
    create_figure_2_main_results()
    print("✅ Figure 2: Main results created")
    
    create_figure_3_comprehensive_results()
    print("✅ Figure 3: Comprehensive results created")
    
    create_supplementary_trajectories()
    print("✅ Supplementary Figure: Trajectories created")
    
    print("\n📊 All publication figures saved to comprehensive_results/")
