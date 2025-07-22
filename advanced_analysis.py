#!/usr/bin/env python3
"""
Advanced analysis strategies for running training study.
Multi-level and time-series approaches.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

def growth_curve_analysis(df, metric):
    """
    Fit growth curves to model individual trajectories.
    Better captures individual differences in training response.
    """
    
    # Prepare data with centered time
    df_growth = df.copy()
    df_growth['time_centered'] = df_growth['timepoint'] - 2.5  # Center at mid-point
    df_growth['time_squared'] = df_growth['time_centered'] ** 2
    
    # Fit hierarchical growth model
    formula = f"{metric} ~ time_centered + time_squared + (time_centered | subject)"
    
    try:
        model = smf.mixedlm(formula, df_growth, groups="subject")
        result = model.fit()
        
        # Extract individual slopes
        random_effects = result.random_effects
        individual_slopes = {}
        
        for subject in df_growth['subject'].unique():
            if subject in random_effects:
                slope = result.params['time_centered'] + random_effects[subject].get('time_centered', 0)
                individual_slopes[subject] = slope
        
        return {
            'model_summary': result.summary(),
            'individual_slopes': individual_slopes,
            'group_linear_slope': result.params.get('time_centered', np.nan),
            'group_quadratic': result.params.get('time_squared', np.nan),
            'linear_p': result.pvalues.get('time_centered', np.nan),
            'quadratic_p': result.pvalues.get('time_squared', np.nan)
        }
        
    except Exception as e:
        print(f"Growth curve analysis failed: {e}")
        return None

def change_point_analysis(df, metric):
    """
    Detect when training effects begin (change point analysis).
    Tests if there's a specific timepoint where change accelerates.
    """
    
    results = {}
    
    # Test different change points
    for change_point in [2, 3]:  # Session 2 or 3 as potential change points
        df_cp = df.copy()
        
        # Create change point variables
        df_cp['before_change'] = np.where(df_cp['timepoint'] <= change_point, 
                                         df_cp['timepoint'] - 1, change_point - 1)
        df_cp['after_change'] = np.where(df_cp['timepoint'] > change_point, 
                                        df_cp['timepoint'] - change_point, 0)
        
        # Fit piecewise model
        formula = f"{metric} ~ before_change + after_change + (1|subject)"
        
        try:
            model = smf.mixedlm(formula, df_cp, groups="subject")
            result = model.fit()
            
            results[f'change_at_{change_point}'] = {
                'before_slope': result.params.get('before_change', np.nan),
                'after_slope': result.params.get('after_change', np.nan),
                'slope_difference': result.params.get('after_change', 0) - result.params.get('before_change', 0),
                'aic': result.aic,
                'before_p': result.pvalues.get('before_change', np.nan),
                'after_p': result.pvalues.get('after_change', np.nan)
            }
            
        except Exception as e:
            print(f"Change point analysis at {change_point} failed: {e}")
    
    return results

def dose_response_detailed(df, metric):
    """
    Detailed dose-response analysis with different training models.
    """
    
    # Create training dose variables
    df_dose = df.copy()
    
    # Cumulative training dose (sessions completed)
    df_dose['training_dose'] = np.where(df_dose['timepoint'] <= 2, 0, df_dose['timepoint'] - 2)
    
    # Exponential decay model (recent training weighted more)
    decay_weights = [0.5, 1.0]  # Weights for sessions 3 and 4
    df_dose['weighted_dose'] = 0
    for i, tp in enumerate([3, 4]):
        df_dose.loc[df_dose['timepoint'] == tp, 'weighted_dose'] = decay_weights[i-1] if i > 0 else 0
    
    models = {}
    
    # Linear dose-response
    try:
        formula = f"{metric} ~ training_dose + (1|subject)"
        model = smf.mixedlm(formula, df_dose, groups="subject")
        result = model.fit()
        
        models['linear_dose'] = {
            'slope': result.params.get('training_dose', np.nan),
            'p_value': result.pvalues.get('training_dose', np.nan),
            'aic': result.aic
        }
    except Exception as e:
        print(f"Linear dose model failed: {e}")
    
    # Threshold model (effect only after minimum dose)
    try:
        df_dose['threshold_dose'] = np.where(df_dose['training_dose'] >= 1, df_dose['training_dose'] - 1, 0)
        formula = f"{metric} ~ threshold_dose + (1|subject)"
        model = smf.mixedlm(formula, df_dose, groups="subject")
        result = model.fit()
        
        models['threshold_dose'] = {
            'slope': result.params.get('threshold_dose', np.nan),
            'p_value': result.pvalues.get('threshold_dose', np.nan),
            'aic': result.aic
        }
    except Exception as e:
        print(f"Threshold dose model failed: {e}")
    
    return models

def individual_response_analysis(df, metric):
    """
    Classify individuals as responders vs non-responders.
    """
    
    # Calculate individual training effects
    individual_effects = []
    
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject].sort_values('timepoint')
        
        if len(subj_data) >= 4:
            # Training effect (T4 - T2)
            training_effect = subj_data.iloc[3][metric] - subj_data.iloc[1][metric]
            
            # Control effect (T2 - T1)  
            control_effect = subj_data.iloc[1][metric] - subj_data.iloc[0][metric]
            
            # Net training effect
            net_effect = training_effect - control_effect
            
            individual_effects.append({
                'subject': subject,
                'training_effect': training_effect,
                'control_effect': control_effect,
                'net_effect': net_effect,
                'baseline_value': subj_data.iloc[0][metric]
            })
    
    effects_df = pd.DataFrame(individual_effects)
    
    # Define responders (e.g., top 50% of net effects, or effect > 1 SD)
    effect_threshold = effects_df['net_effect'].std()
    effects_df['responder'] = effects_df['net_effect'] > effect_threshold
    
    n_responders = effects_df['responder'].sum()
    n_total = len(effects_df)
    
    # Test if responder rate is above chance
    from scipy.stats import binom_test
    p_responder = binom_test(n_responders, n_total, 0.5, alternative='greater')
    
    # Analyze responder characteristics
    responder_baseline = effects_df[effects_df['responder']]['baseline_value'].mean()
    non_responder_baseline = effects_df[~effects_df['responder']]['baseline_value'].mean()
    
    baseline_diff_p = stats.ttest_ind(
        effects_df[effects_df['responder']]['baseline_value'],
        effects_df[~effects_df['responder']]['baseline_value']
    )[1]
    
    return {
        'responder_rate': n_responders / n_total,
        'p_value_responder_rate': p_responder,
        'responder_baseline_mean': responder_baseline,
        'non_responder_baseline_mean': non_responder_baseline,
        'baseline_difference_p': baseline_diff_p,
        'individual_effects': effects_df
    }

def network_trajectory_clustering(df, metrics):
    """
    Cluster subjects based on their network trajectory patterns.
    Identifies different patterns of training response.
    """
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Calculate trajectory features for each subject
    trajectory_features = []
    
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject].sort_values('timepoint')
        
        if len(subj_data) >= 4:
            features = {}
            
            for metric in metrics:
                # Calculate various trajectory features
                values = subj_data[metric].values
                
                features[f'{metric}_baseline'] = values[0]
                features[f'{metric}_final'] = values[3]
                features[f'{metric}_total_change'] = values[3] - values[0]
                features[f'{metric}_control_change'] = values[1] - values[0]
                features[f'{metric}_training_change'] = values[3] - values[1]
                features[f'{metric}_variability'] = np.std(values)
                
                # Linear trend
                x = np.arange(4)
                slope = np.polyfit(x, values, 1)[0]
                features[f'{metric}_slope'] = slope
            
            features['subject'] = subject
            trajectory_features.append(features)
    
    features_df = pd.DataFrame(trajectory_features)
    
    # Standardize features
    feature_cols = [col for col in features_df.columns if col != 'subject']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df[feature_cols])
    
    # Cluster analysis (try 2-4 clusters)
    cluster_results = {}
    
    for n_clusters in [2, 3, 4]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        features_df[f'cluster_{n_clusters}'] = clusters
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(features_scaled, clusters)
        
        cluster_results[n_clusters] = {
            'silhouette_score': sil_score,
            'cluster_sizes': np.bincount(clusters)
        }
    
    return features_df, cluster_results

if __name__ == "__main__":
    # Example usage
    print("Advanced analysis strategies for running training study")
    print("Run these functions with your loaded data for additional insights!")
