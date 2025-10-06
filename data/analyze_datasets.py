#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes all downloaded datasets to understand their structure,
data types, missing values, and preprocessing requirements.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_dataset(file_path, dataset_name):
    """Analyze a single dataset and return summary information."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_name}")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Basic information
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Data types
        print(f"\nData Types:")
        print(df.dtypes)
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found.")
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"\nCategorical columns: {categorical_cols}")
            for col in categorical_cols:
                print(f"  {col}: {df[col].nunique()} unique values")
                if df[col].nunique() <= 10:
                    print(f"    Values: {df[col].unique()}")
        else:
            print("\nNo categorical columns found.")
        
        # Numerical summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            print(f"\nNumerical columns: {len(numerical_cols)}")
            print("\nNumerical summary:")
            print(df[numerical_cols].describe())
            
            # Check for potential outliers using IQR method
            print(f"\nPotential outliers (using IQR method):")
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
        
        # Identify potential target variable
        # Look for columns with names suggesting they're targets
        target_candidates = []
        target_keywords = ['target', 'price', 'quality', 'strength', 'rings', 'cnt', 'medv', 'level']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_candidates.append(col)
        
        if target_candidates:
            print(f"\nPotential target variables: {target_candidates}")
        else:
            # If no obvious target, suggest the last column
            print(f"\nNo obvious target variable found. Last column might be target: '{df.columns[-1]}'")
        
        return {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': missing.to_dict(),
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'target_candidates': target_candidates if target_candidates else [df.columns[-1]],
            'has_missing': missing.sum() > 0,
            'has_categorical': len(categorical_cols) > 0
        }
        
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        return None

def main():
    """Main function to analyze all datasets."""
    print("DATASET ANALYSIS REPORT")
    print("="*80)
    
    # Define dataset paths
    datasets = {
        'California Housing': 'california_housing/california_housing.csv',
        'Diabetes': 'diabetes/diabetes.csv',
        'Energy Efficiency': 'energy_efficiency/energy_efficiency.csv',
        'Concrete Strength': 'concrete_strength/concrete_strength.csv',
        'Wine Quality Red': 'wine_quality/wine_quality_red.csv',
        'Wine Quality White': 'wine_quality/wine_quality_white.csv',
        'Wine Quality Combined': 'wine_quality/wine_quality_combined.csv',
        'Airfoil Self-Noise': 'airfoil_noise/airfoil_self_noise.csv',
        'Bike Sharing Day': 'bike_sharing/day.csv',
        'Bike Sharing Hour': 'bike_sharing/hour.csv',
        'Abalone': 'abalone/abalone.csv'
    }
    
    analysis_results = {}
    
    for name, path in datasets.items():
        if os.path.exists(path):
            result = analyze_dataset(path, name)
            if result:
                analysis_results[name] = result
        else:
            print(f"\nWarning: {path} not found!")
    
    # Summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    print(f"\nTotal datasets analyzed: {len(analysis_results)}")
    
    # Datasets with missing values
    missing_datasets = [name for name, info in analysis_results.items() if info['has_missing']]
    if missing_datasets:
        print(f"\nDatasets with missing values: {missing_datasets}")
    else:
        print(f"\nNo datasets have missing values.")
    
    # Datasets with categorical variables
    categorical_datasets = [name for name, info in analysis_results.items() if info['has_categorical']]
    if categorical_datasets:
        print(f"\nDatasets with categorical variables: {categorical_datasets}")
    else:
        print(f"\nNo datasets have categorical variables.")
    
    # Dataset sizes
    print(f"\nDataset sizes:")
    for name, info in analysis_results.items():
        print(f"  {name}: {info['shape'][0]} rows, {info['shape'][1]} columns")
    
    print(f"\n{'='*80}")
    print("PREPROCESSING RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    for name, info in analysis_results.items():
        print(f"\n{name}:")
        
        recommendations = []
        if info['has_missing']:
            recommendations.append("- Handle missing values (imputation)")
        if info['has_categorical']:
            recommendations.append("- Encode categorical variables (one-hot encoding)")
        recommendations.append("- Standardize numerical features")
        recommendations.append("- Center data for PCA")
        recommendations.append("- Check and handle outliers")
        recommendations.append("- Create train/test splits")
        
        for rec in recommendations:
            print(f"  {rec}")

if __name__ == "__main__":
    main()