import pandas as pd
import numpy as np
from scipy import stats
import os

# Load the data
df = pd.read_csv('graph_metrics_global.csv')
print(f"📊 Loaded data: {df.shape}")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Timepoints: {sorted(df['timepoint'].unique())}")

# Simple contrast analysis for global_efficiency
metric = 'global_efficiency'
print(f"\n🧠 Analyzing: {metric}")

# Calculate changes for each subject
results = []
for subject in df['subject'].unique():
    subj_data = df[df['subject'] == subject].sort_values('timepoint')
    if len(subj_data) >= 4:
        # Control period change (T2 - T1)
        control_change = subj_data.iloc[1][metric] - subj_data.iloc[0][metric]
        # Training period change (T4 - T2)
        training_change = subj_data.iloc[3][metric] - subj_data.iloc[1][metric]
        
        results.append({
            'subject': subject,
            'control_change': control_change,
            'training_change': training_change,
            'net_effect': training_change - control_change
        })

results_df = pd.DataFrame(results)
print(f"\n📈 Results for {len(results_df)} subjects:")
print(f"Mean control change: {results_df['control_change'].mean():.6f}")
print(f"Mean training change: {results_df['training_change'].mean():.6f}")
print(f"Mean net training effect: {results_df['net_effect'].mean():.6f}")

# Statistical test
t_stat, p_value = stats.ttest_1samp(results_df['net_effect'], 0)
print(f"\n🎯 Statistical test:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value < 0.05:
    print("✅ SIGNIFICANT training effect!")
else:
    print("❌ No significant training effect")

# Save results
os.makedirs('simple_results', exist_ok=True)
results_df.to_csv('simple_results/training_effects.csv', index=False)
print(f"\n💾 Results saved to: simple_results/training_effects.csv")
