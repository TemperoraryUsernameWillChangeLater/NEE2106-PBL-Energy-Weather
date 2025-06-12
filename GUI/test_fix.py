#!/usr/bin/env python3
"""
Test script to verify the encoding fix for temperature column matching
"""
import pandas as pd

def normalize_column_name(col_name):
    """Normalize column names to handle encoding artifacts"""
    return str(col_name).lower().replace('ï¿½', 'o').replace('°', 'o')

# Test the fix
print("TESTING ENCODING FIX FOR TEMPERATURE COLUMNS")
print("=" * 50)

# Load a sample dataset that has the encoding issue (March 2025)
try:
    df_march = pd.read_csv('../Datasets/25-03.csv', skiprows=7, encoding='latin-1')
    print(f"✓ Loaded March 2025: {len(df_march)} rows")
    
    # Show actual column headers
    temp_cols = [col for col in df_march.columns if 'temperature' in str(col).lower()]
    print(f"Temperature columns in March 2025: {temp_cols}")
    
    # Test old vs new column matching
    print("\nTESTING COLUMN MATCHING:")
    print("-" * 30)
    
    variable = "Max Temperature"
    
    # OLD METHOD (fails)
    col_name_old = None
    for col in df_march.columns:
        if "maximum temperature" in col.lower():
            col_name_old = col
            break
    
    # NEW METHOD (should work)
    col_name_new = None
    for col in df_march.columns:
        if "maximum temperature" in normalize_column_name(col):
            col_name_new = col
            break
    
    print(f"OLD method found: {col_name_old}")
    print(f"NEW method found: {col_name_new}")
    
    if col_name_new:
        clean_data = pd.to_numeric(df_march[col_name_new], errors='coerce').dropna()
        print(f"✓ NEW method can extract {len(clean_data)} temperature values")
        print(f"  Temperature range: {clean_data.min():.1f} to {clean_data.max():.1f}")
    else:
        print("✗ NEW method failed to find column")
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    if col_name_old is None and col_name_new is not None:
        print("✅ FIX SUCCESSFUL! New method finds March temperature data.")
        print("   This should resolve the missing 31 days issue.")
    else:
        print("❌ Fix may not be working as expected.")
        
except Exception as e:
    print(f"❌ Error: {e}")
