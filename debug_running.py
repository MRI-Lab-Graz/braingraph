#!/usr/bin/env python3
import os
import pandas as pd
import json

print("🔍 DEBUGGING RUNNING ANALYSIS")
print("="*40)

# Check current directory
print(f"📁 Current directory: {os.getcwd()}")

# Check for CSV files
print("\n📊 Looking for CSV files...")
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    print("Found CSV files:")
    for f in csv_files:
        print(f"  - {f}")
else:
    print("❌ No CSV files found!")

# Check specific file from config
config_file = "running_config.json"
if os.path.exists(config_file):
    print(f"\n✅ Found config file: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    input_file = config.get('input_file')
    print(f"📄 Config specifies input file: {input_file}")
    
    if os.path.exists(input_file):
        print(f"✅ Input file exists: {input_file}")
        
        # Try to load the data
        try:
            df = pd.read_csv(input_file)
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check for required columns
            if 'subject' in df.columns and 'timepoint' in df.columns:
                print(f"\n📊 Data summary:")
                print(f"   Subjects: {df['subject'].nunique()}")
                print(f"   Timepoints: {sorted(df['timepoint'].unique())}")
                print(f"   Total observations: {len(df)}")
            else:
                print("❌ Missing required columns (subject, timepoint)")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    else:
        print(f"❌ Input file not found: {input_file}")
else:
    print(f"❌ Config file not found: {config_file}")

print(f"\n🚀 Suggested commands:")
if csv_files:
    print(f"python running_analysis.py --data {csv_files[0]}")
print(f"python running_analysis.py --config running_config.json")
