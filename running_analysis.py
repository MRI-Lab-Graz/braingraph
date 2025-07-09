#!/usr/bin/env python3
"""
Advanced statistical analysis for running training effects on brain graph metrics.

This script implements multiple statistical approaches tailored to the specific
experimental design: control period (T1→T2) vs training period (T2→T4).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import os
import json
import argparse
from datetime import datetime

def load_and_prepare_data(file_path):
    """Load data and create derived variables for running analysis."""
    df = pd.read_csv(file_path)
    df['subject'] = df['subject'].astype(str)
    df['timepoint'] = df['timepoint'].astype(int)
    
    # Sort by subject and timepoint
    df = df.sort_values(['subject', 'timepoint']).reset_index(drop=True)
    
    # Create period variable
    df['period'] = df['timepoint'].map({
        1: 'baseline',
        2: 'pre_training', 
        3: 'mid_training',
        4: 'post_training'
    })
    
    # Create training phase (control vs training)
    df['training_phase'] = df['timepoint'].map({
        1: 'baseline',
        2: 'control',
        3: 'training', 
        4: 'training'
    })
    
    # Create time variables for piecewise analysis
    df['time_overall'] = df['timepoint'] - 1  # 0, 1, 2, 3
    df['time_in_control'] = np.where(df['timepoint'] <= 2, df['timepoint'] - 1, 1)
    df['time_in_training'] = np.where(df['timepoint'] > 2, df['timepoint'] - 2, 0)
    
    # Create change scores
    metrics = [col for col in df.columns if col not in ['subject', 'timepoint', 'period', 'training_phase', 'time_overall', 'time_in_control', 'time_in_training']]
    
    for metric in metrics:
        # Calculate change from baseline
        df[f'{metric}_change_from_baseline'] = df.groupby('subject')[metric].transform(lambda x: x - x.iloc[0])
        
        # Calculate change from previous timepoint
        df[f'{metric}_change_from_prev'] = df.groupby('subject')[metric].diff()
    
    return df, metrics

def contrast_analysis(df, metric, output_dir):
    """Perform contrast-based analysis comparing control vs training periods."""
    print(f"\n=== CONTRAST ANALYSIS: {metric} ===")
    
    # Calculate period-specific changes
    changes = []
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject].sort_values('timepoint')
        
        if len(subj_data) >= 4:  # Ensure we have all timepoints
            # Control period change (T2 - T1)
            control_change = subj_data.iloc[1][metric] - subj_data.iloc[0][metric]
            
            # Training period change (T4 - T2)
            training_change = subj_data.iloc[3][metric] - subj_data.iloc[1][metric]
            
            # Mid-training change (T3 - T2) for dose-response
            mid_training_change = subj_data.iloc[2][metric] - subj_data.iloc[1][metric]
            
            changes.append({
                'subject': subject,
                'control_change': control_change,
                'training_change': training_change,
                'mid_training_change': mid_training_change,
                'training_vs_control': training_change - control_change,
                'baseline_value': subj_data.iloc[0][metric]
            })
    
    changes_df = pd.DataFrame(changes)
    
    # Statistical tests
    results = {}
    
    # 1. Paired t-test: Training change vs Control change
    t_stat, p_val = stats.ttest_rel(changes_df['training_change'], changes_df['control_change'])
    results['training_vs_control'] = {
        'test': 'Paired t-test (Training vs Control)',
        't_statistic': t_stat,
        'p_value': p_val,
        'effect_size': np.mean(changes_df['training_vs_control']) / np.std(changes_df['training_vs_control']),
        'mean_diff': np.mean(changes_df['training_vs_control']),
        'std_diff': np.std(changes_df['training_vs_control'])
    }
    
    # 2. One-sample t-test: Is training effect different from zero?
    t_stat, p_val = stats.ttest_1samp(changes_df['training_change'], 0)
    results['training_effect'] = {
        'test': 'One-sample t-test (Training effect)',
        't_statistic': t_stat,
        'p_value': p_val,
        'mean_change': np.mean(changes_df['training_change']),
        'std_change': np.std(changes_df['training_change'])
    }
    
    # 3. Control for baseline values
    model_baseline = smf.ols('training_vs_control ~ baseline_value', data=changes_df).fit()
    results['baseline_adjusted'] = {
        'test': 'Baseline-adjusted training effect',
        'coefficient': model_baseline.params['baseline_value'],
        'p_value': model_baseline.pvalues['baseline_value'],
        'r_squared': model_baseline.rsquared
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Running Training Analysis: {metric}', fontsize=16)
    
    # Plot 1: Individual trajectories
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject].sort_values('timepoint')
        axes[0,0].plot(subj_data['timepoint'], subj_data[metric], 'o-', alpha=0.3, color='gray')
    
    # Add mean trajectory
    mean_trajectory = df.groupby('timepoint')[metric].mean()
    axes[0,0].plot(mean_trajectory.index, mean_trajectory.values, 'o-', color='red', linewidth=3, label='Mean')
    axes[0,0].axvline(x=2, color='blue', linestyle='--', alpha=0.7, label='Training Start')
    axes[0,0].set_xlabel('Session')
    axes[0,0].set_ylabel(metric)
    axes[0,0].set_title('Individual Trajectories')
    axes[0,0].legend()
    
    # Plot 2: Change comparison
    x_pos = [1, 2]
    changes_mean = [np.mean(changes_df['control_change']), np.mean(changes_df['training_change'])]
    changes_sem = [stats.sem(changes_df['control_change']), stats.sem(changes_df['training_change'])]
    
    axes[0,1].bar(x_pos, changes_mean, yerr=changes_sem, capsize=5, 
                  color=['lightblue', 'orange'], alpha=0.7)
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(['Control Period\n(T2-T1)', 'Training Period\n(T4-T2)'])
    axes[0,1].set_ylabel(f'Change in {metric}')
    axes[0,1].set_title(f'Period Comparison (p={results["training_vs_control"]["p_value"]:.3f})')
    
    # Plot 3: Training vs Control scatter
    axes[1,0].scatter(changes_df['control_change'], changes_df['training_change'], alpha=0.7)
    axes[1,0].plot([changes_df['control_change'].min(), changes_df['control_change'].max()], 
                   [changes_df['control_change'].min(), changes_df['control_change'].max()], 
                   'r--', alpha=0.5, label='No difference line')
    axes[1,0].set_xlabel('Control Period Change')
    axes[1,0].set_ylabel('Training Period Change')
    axes[1,0].set_title('Training vs Control Changes')
    axes[1,0].legend()
    
    # Plot 4: Effect size distribution
    axes[1,1].hist(changes_df['training_vs_control'], bins=10, alpha=0.7, color='green')
    axes[1,1].axvline(x=0, color='red', linestyle='--', label='No effect')
    axes[1,1].axvline(x=np.mean(changes_df['training_vs_control']), color='blue', 
                      linewidth=2, label=f'Mean effect')
    axes[1,1].set_xlabel('Training - Control Change')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Effect Size Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_running_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return results, changes_df

def piecewise_analysis(df, metric):
    """Perform piecewise regression analysis."""
    print(f"\n=== PIECEWISE ANALYSIS: {metric} ===")
    
    # Model with different slopes for control and training periods
    formula = f"{metric} ~ time_in_control + time_in_training + (1|subject)"
    
    try:
        model = smf.mixedlm(formula, df, groups="subject")
        result = model.fit()
        
        return {
            'model': 'Piecewise Mixed Effects',
            'control_slope': result.params.get('time_in_control', np.nan),
            'control_p': result.pvalues.get('time_in_control', np.nan),
            'training_slope': result.params.get('time_in_training', np.nan),
            'training_p': result.pvalues.get('time_in_training', np.nan),
            'slope_difference': result.params.get('time_in_training', np.nan) - result.params.get('time_in_control', np.nan),
            'aic': result.aic,
            'summary': result.summary()
        }
    except Exception as e:
        print(f"Piecewise analysis failed: {e}")
        return None

def dose_response_analysis(df, metric):
    """Analyze dose-response relationship in training period."""
    print(f"\n=== DOSE-RESPONSE ANALYSIS: {metric} ===")
    
    # Focus on training period (sessions 2, 3, 4)
    training_data = df[df['timepoint'] >= 2].copy()
    training_data['training_time'] = training_data['timepoint'] - 2  # 0, 1, 2
    
    formula = f"{metric} ~ training_time + (1|subject)"
    
    try:
        model = smf.mixedlm(formula, training_data, groups="subject")
        result = model.fit()
        
        return {
            'model': 'Dose-Response (Training Period)',
            'linear_slope': result.params.get('training_time', np.nan),
            'linear_p': result.pvalues.get('training_time', np.nan),
            'aic': result.aic
        }
    except Exception as e:
        print(f"Dose-response analysis failed: {e}")
        return None

def generate_power_analysis_recommendations(df, metric, results, output_dir):
    """Generate post-hoc power analysis and sample size recommendations."""
    try:
        from statsmodels.stats.power import ttest_power
        
        changes = []
        for subject in df['subject'].unique():
            subj_data = df[df['subject'] == subject].sort_values('timepoint')
            if len(subj_data) >= 4:
                control_change = subj_data.iloc[1][metric] - subj_data.iloc[0][metric]
                training_change = subj_data.iloc[3][metric] - subj_data.iloc[1][metric]
                changes.append({
                    'control_change': control_change,
                    'training_change': training_change,
                    'difference': training_change - control_change
                })
        
        changes_df = pd.DataFrame(changes)
        
        # Calculate observed effect size
        observed_d = results['training_vs_control']['effect_size']
        current_n = len(changes_df)
        
        # Post-hoc power calculation
        observed_power = ttest_power(observed_d, current_n, alpha=0.05)
        
        # Sample size recommendations for future studies
        power_80_n = ttest_power(observed_d, nobs=None, alpha=0.05, power=0.80)
        power_90_n = ttest_power(observed_d, nobs=None, alpha=0.05, power=0.90)
        
        power_report = {
            'observed_effect_size': observed_d,
            'current_sample_size': current_n,
            'observed_power': observed_power,
            'n_for_80_power': power_80_n,
            'n_for_90_power': power_90_n
        }
        
        # Save power analysis
        with open(os.path.join(output_dir, f"{metric}_power_analysis.txt"), "w") as f:
            f.write(f"POWER ANALYSIS: {metric}\n")
            f.write("="*40 + "\n\n")
            f.write(f"Observed effect size (Cohen's d): {observed_d:.4f}\n")
            f.write(f"Current sample size: {current_n}\n")
            f.write(f"Observed power: {observed_power:.4f} ({observed_power*100:.1f}%)\n\n")
            f.write("SAMPLE SIZE RECOMMENDATIONS:\n")
            f.write(f"For 80% power: {power_80_n:.0f} subjects\n")
            f.write(f"For 90% power: {power_90_n:.0f} subjects\n\n")
            
            if observed_power < 0.80:
                f.write("⚠️ Current study may be underpowered (power < 80%)\n")
            else:
                f.write("✅ Current study appears adequately powered\n")
        
        return power_report
        
    except ImportError:
        print("⚠️ statsmodels.stats.power not available for power analysis")
        return None
    except Exception as e:
        print(f"⚠️ Power analysis failed: {e}")
        return None

def create_effect_size_summary_plot(all_results, output_dir):
    """Create a summary plot showing effect sizes across all metrics."""
    
    metrics = []
    effect_sizes = []
    p_values = []
    
    for metric, results in all_results.items():
        if 'contrast' in results:
            metrics.append(metric.replace('_', '\n'))
            effect_sizes.append(results['contrast']['training_vs_control']['effect_size'])
            p_values.append(results['contrast']['training_vs_control']['p_value'])
    
    if not metrics:
        return
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Effect sizes plot
    colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
    bars = ax1.barh(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics)
    ax1.set_xlabel("Effect Size (Cohen's d)")
    ax1.set_title("Training Effect Sizes by Metric")
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
    ax1.legend()
    
    # Add effect size values
    for i, (effect, p) in enumerate(zip(effect_sizes, p_values)):
        ax1.text(effect + 0.02, i, f'{effect:.3f}', va='center')
    
    # P-values plot (negative log scale)
    neg_log_p = [-np.log10(p) for p in p_values]
    bars2 = ax2.barh(range(len(metrics)), neg_log_p, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels(metrics)
    ax2.set_xlabel("-log10(p-value)")
    ax2.set_title("Statistical Significance by Metric")
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axvline(x=-np.log10(0.01), color='red', linestyle='--', alpha=0.9, label='p = 0.01')
    ax2.legend()
    
    # Add p-values
    for i, p in enumerate(p_values):
        ax2.text(-np.log10(p) + 0.02, i, f'{p:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_effects_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Effect size summary plot saved")

def run_config_models(df, metric, model_formulas, primary_model, output_dir):
    """Run the model formulas specified in the config file."""
    print(f"\n=== CONFIG-BASED MODEL ANALYSIS: {metric} ===")
    
    results = {}
    
    # Create derived variables needed for the models
    df_model = df.copy()
    df_model['post_training'] = (df_model['timepoint'] > 2).astype(int)
    df_model['time_in_control'] = np.where(df_model['timepoint'] <= 2, df_model['timepoint'] - 1, 1)
    df_model['time_in_training'] = np.where(df_model['timepoint'] > 2, df_model['timepoint'] - 2, 0)
    
    # Add baseline values for baseline_adjusted model
    baseline_values = df_model[df_model['timepoint'] == 1].groupby('subject')[metric].first()
    df_model['baseline_value'] = df_model['subject'].map(baseline_values)
    
    for model_name, formula_template in model_formulas.items():
        print(f"\n--- Running {model_name} model ---")
        
        try:
            # Insert metric name into formula
            formula = formula_template.format(metric=metric)
            print(f"Formula: {formula}")
            
            # Run mixed-effects model
            model = smf.mixedlm(formula, df_model, groups="subject")
            result = model.fit()
            
            # Save detailed results
            model_results = {
                'formula': formula,
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.llf,
                'converged': result.converged,
                'params': dict(result.params),
                'pvalues': dict(result.pvalues),
                'confidence_intervals': result.conf_int().to_dict(),
                'summary': result.summary().as_text()
            }
            
            results[model_name] = model_results
            
            # Save individual model summary
            summary_path = os.path.join(output_dir, f"{metric}_{model_name}_model.txt")
            with open(summary_path, "w") as f:
                f.write(f"MODEL: {model_name}\n")
                f.write(f"FORMULA: {formula}\n")
                f.write("="*60 + "\n")
                f.write(result.summary().as_text())
                
                # Add interpretation for key models
                if model_name == "pre_post_contrast" and 'post_training' in result.params:
                    coef = result.params['post_training']
                    p_val = result.pvalues['post_training']
                    f.write(f"\n\nINTERPRETATION:\n")
                    f.write(f"Training effect: {coef:.6f}\n")
                    f.write(f"P-value: {p_val:.4f}\n")
                    f.write(f"Significance: {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}\n")
            
            print(f"✔️ {model_name} model completed (AIC: {result.aic:.2f})")
            
        except Exception as e:
            print(f"❌ {model_name} model failed: {e}")
            results[model_name] = {'error': str(e)}
    
    # Identify best model by AIC
    valid_models = {k: v for k, v in results.items() if 'aic' in v}
    if valid_models:
        best_model = min(valid_models.keys(), key=lambda k: valid_models[k]['aic'])
        print(f"\n🏆 Best model by AIC: {best_model} (AIC: {valid_models[best_model]['aic']:.2f})")
    
    # Highlight primary model results
    if primary_model in results and 'error' not in results[primary_model]:
        primary_results = results[primary_model]
        print(f"\n🎯 PRIMARY MODEL ({primary_model}) RESULTS:")
        for param, value in primary_results['params'].items():
            if param != 'Intercept':
                p_val = primary_results['pvalues'][param]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                print(f"  {param}: {value:.6f} (p = {p_val:.4f}) {sig}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Advanced running training analysis")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--data", type=str, help="Path to graph metrics CSV (overrides config)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config)")
    parser.add_argument("--metrics", nargs="+", help="Specific metrics to analyze (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Command line arguments override config file
    data_file = args.data or config.get("input_file", config.get("data_file"))
    output_dir = args.output or config.get("output_folder", config.get("output_dir", "running_analysis_results"))
    metrics_list = args.metrics or config.get("metrics")
    
    # Get model formulas from config
    model_formulas = config.get("model_formulas", {})
    primary_model = config.get("primary_model", "pre_post_contrast")
    
    if not data_file:
        parser.error("Either --data argument or 'input_file'/'data_file' in config file is required")
    
    # Update args for backward compatibility
    args.data = data_file
    args.output = output_dir
    args.metrics = metrics_list
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "plots"), exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    df, available_metrics = load_and_prepare_data(args.data)
    
    # Select metrics to analyze
    if args.metrics:
        metrics = [m for m in args.metrics if m in available_metrics]
    else:
        metrics = available_metrics
    
    print(f"Analyzing {len(metrics)} metrics for {df['subject'].nunique()} subjects")
    print(f"Sessions per subject: {df.groupby('subject').size().describe()}")
    
    # Run analyses for each metric
    all_results = {}
    
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {metric}")
        print(f"{'='*60}")
        
        metric_results = {}
        
        # 1. Config-based model analysis (if model formulas provided)
        if model_formulas:
            config_results = run_config_models(df, metric, model_formulas, primary_model, args.output)
            metric_results['config_models'] = config_results
        
        # 2. Contrast analysis (built-in approach)
        contrast_results, changes_df = contrast_analysis(df, metric, os.path.join(args.output, "plots"))
        metric_results['contrast'] = contrast_results
        
        # Save individual changes
        changes_df.to_csv(os.path.join(args.output, f"{metric}_changes.csv"), index=False)
        
        # 3. Piecewise analysis (built-in)
        piecewise_results = piecewise_analysis(df, metric)
        if piecewise_results:
            metric_results['piecewise'] = piecewise_results
        
        # 4. Dose-response analysis (built-in)
        dose_results = dose_response_analysis(df, metric)
        if dose_results:
            metric_results['dose_response'] = dose_results
        
        # 5. Power analysis
        power_results = generate_power_analysis_recommendations(df, metric, contrast_results, args.output)
        if power_results:
            metric_results['power'] = power_results
        
        # 5. Config-based model analysis
        config_results = run_config_models(df, metric, model_formulas, primary_model, args.output)
        if config_results:
            metric_results['config_models'] = config_results
        
        all_results[metric] = metric_results
        
        # Print summary
        print(f"\nSUMMARY for {metric}:")
        print(f"Training vs Control effect: {contrast_results['training_vs_control']['mean_diff']:.4f}")
        print(f"P-value: {contrast_results['training_vs_control']['p_value']:.4f}")
        print(f"Effect size (Cohen's d): {contrast_results['training_vs_control']['effect_size']:.4f}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary report
    with open(os.path.join(args.output, f"running_analysis_summary_{timestamp}.txt"), "w") as f:
        f.write("RUNNING TRAINING EFFECT ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Number of subjects: {df['subject'].nunique()}\n")
        f.write(f"Number of metrics: {len(metrics)}\n\n")
        
        f.write("EXPERIMENTAL DESIGN:\n")
        f.write("- Session 1: Baseline\n")
        f.write("- Session 2: Pre-training (end of control period)\n") 
        f.write("- Session 3: Mid-training\n")
        f.write("- Session 4: Post-training\n")
        f.write("- Control period: Session 1 → Session 2 (no training)\n")
        f.write("- Training period: Session 2 → Session 4 (running training)\n\n")
        
        f.write("MAIN FINDINGS:\n")
        f.write("-" * 30 + "\n")
        
        significant_effects = []
        for metric, results in all_results.items():
            p_val = results['contrast']['training_vs_control']['p_value']
            effect_size = results['contrast']['training_vs_control']['effect_size']
            mean_diff = results['contrast']['training_vs_control']['mean_diff']
            
            f.write(f"\n{metric.upper()}:\n")
            f.write(f"  Training vs Control difference: {mean_diff:.6f}\n")
            f.write(f"  P-value: {p_val:.4f}\n")
            f.write(f"  Effect size (Cohen's d): {effect_size:.4f}\n")
            f.write(f"  Significance: {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}\n")
            
            # Add power analysis if available
            if 'power' in results and results['power']:
                power_info = results['power']
                f.write(f"  Observed power: {power_info['observed_power']:.3f} ({power_info['observed_power']*100:.1f}%)\n")
                f.write(f"  Sample size for 80% power: {power_info['n_for_80_power']:.0f}\n")
            
            if p_val < 0.05:
                significant_effects.append((metric, p_val, effect_size))
        
        if significant_effects:
            f.write(f"\nSIGNIFICANT TRAINING EFFECTS (p < 0.05):\n")
            for metric, p_val, effect_size in sorted(significant_effects, key=lambda x: x[1]):
                f.write(f"- {metric}: p = {p_val:.4f}, d = {effect_size:.3f}\n")
        else:
            f.write(f"\nNo significant training effects found (all p > 0.05)\n")
        
        # Add power analysis summary
        f.write(f"\nPOWER ANALYSIS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for metric, results in all_results.items():
            if 'power' in results and results['power']:
                power = results['power']['observed_power']
                f.write(f"{metric}: {power:.3f} ({power*100:.1f}%)\n")
    
    # Create effect size summary plot
    create_effect_size_summary_plot(all_results, os.path.join(args.output, "plots"))
    
    print(f"\nAnalysis complete! Results saved to: {args.output}")
    print(f"- Summary report: running_analysis_summary_{timestamp}.txt")
    print(f"- Individual plots: plots/ directory")
    print(f"- Change scores: *_changes.csv files")
    print(f"- Power analyses: *_power_analysis.txt files")
    print(f"- Effect size summary: plots/training_effects_summary.png")
