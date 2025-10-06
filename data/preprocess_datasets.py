#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Script for PCA and Linear Regression
Implements all preprocessing steps according to the specified requirements:
1. Scaling (standardization/z-transformation)
2. Centering (mean=0 for PCA)
3. Categorical variable handling (one-hot encoding)
4. Missing value handling (already checked - none found)
5. Outlier treatment (robust scaling, winsorizing)
6. Dataset splitting (train/test)
7. Target variable preparation
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DatasetPreprocessor:
    """Class to handle comprehensive dataset preprocessing."""
    
    def __init__(self, outlier_method='robust', test_size=0.2, random_state=42):
        """
        Initialize preprocessor with configuration.
        
        Parameters:
        outlier_method: 'robust', 'winsorize', or 'none'
        test_size: proportion for test set
        random_state: for reproducible splits
        """
        self.outlier_method = outlier_method
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        
    def detect_outliers_iqr(self, series, factor=1.5):
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def winsorize_outliers(self, series, limits=(0.05, 0.05)):
        """Apply winsorization to handle outliers."""
        return pd.Series(stats.mstats.winsorize(series, limits=limits), index=series.index)
    
    def handle_categorical_variables(self, df, categorical_cols):
        """Handle categorical variables with appropriate encoding."""
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col == 'dteday':
                # Handle date column - drop it as it's not useful for PCA/regression
                print(f"  Dropping date column: {col}")
                df_processed = df_processed.drop(columns=[col])
            
            elif df[col].nunique() <= 10:  # Low cardinality - use one-hot encoding
                print(f"  One-hot encoding: {col} ({df[col].nunique()} categories)")
                # Create one-hot encoder
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                
                # Store encoder for later use
                self.encoders[col] = encoder
                
                # Replace original column with encoded columns
                df_processed = df_processed.drop(columns=[col])
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            
            else:  # High cardinality - use label encoding
                print(f"  Label encoding: {col} ({df[col].nunique()} categories)")
                encoder = LabelEncoder()
                df_processed[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
        
        return df_processed
    
    def handle_outliers(self, df, numerical_cols):
        """Handle outliers in numerical columns."""
        df_processed = df.copy()
        
        if self.outlier_method == 'robust':
            print("  Using RobustScaler to handle outliers")
            # Will be handled during scaling
            
        elif self.outlier_method == 'winsorize':
            print("  Applying winsorization to outliers")
            for col in numerical_cols:
                if col in df_processed.columns:
                    outliers = self.detect_outliers_iqr(df_processed[col])
                    if outliers.sum() > 0:
                        print(f"    Winsorizing {outliers.sum()} outliers in {col}")
                        df_processed[col] = self.winsorize_outliers(df_processed[col])
        
        elif self.outlier_method == 'none':
            print("  No outlier treatment applied")
        
        return df_processed
    
    def scale_and_center_features(self, X_train, X_test, method='standard'):
        """Apply scaling and centering to features."""
        if method == 'standard':
            scaler = StandardScaler()
            print("  Applying StandardScaler (z-transformation)")
        elif method == 'robust':
            scaler = RobustScaler()
            print("  Applying RobustScaler (robust to outliers)")
        
        # Fit on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Store scaler
        self.scalers['features'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def process_dataset(self, file_path, dataset_name, target_col, categorical_cols=None):
        """
        Complete preprocessing pipeline for a single dataset.
        
        Parameters:
        file_path: path to the CSV file
        dataset_name: name for identification
        target_col: name of target column
        categorical_cols: list of categorical column names
        """
        print(f"\n{'='*60}")
        print(f"PREPROCESSING: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Handle categorical variables
        if categorical_cols:
            print(f"\nStep 3: Handling categorical variables")
            df = self.handle_categorical_variables(df, categorical_cols)
            print(f"Shape after categorical handling: {df.shape}")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nFeatures: {X.shape[1]} columns")
        print(f"Target: {target_col}")
        
        # Get numerical columns (after categorical processing)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Step 5: Handle outliers
        print(f"\nStep 5: Handling outliers")
        X = self.handle_outliers(X, numerical_cols)
        
        # Step 6: Create train/test splits
        print(f"\nStep 6: Creating train/test splits")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Steps 1 & 2: Scale and center features
        print(f"\nSteps 1 & 2: Scaling and centering features")
        scaling_method = 'robust' if self.outlier_method == 'robust' else 'standard'
        X_train_scaled, X_test_scaled = self.scale_and_center_features(
            X_train, X_test, method=scaling_method
        )
        
        # Verify centering (should be close to 0)
        train_means = X_train_scaled.mean().abs().max()
        print(f"Max absolute mean after centering: {train_means:.6f}")
        
        # Create processed dataset dictionary
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': X_train_scaled.columns.tolist(),
            'target_name': target_col,
            'original_shape': df.shape,
            'processed_shape': X_train_scaled.shape,
            'scaler': self.scalers.get('features'),
            'encoders': self.encoders.copy()
        }
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Final feature shape: {X_train_scaled.shape}")
        
        return processed_data

def main():
    """Main function to preprocess all datasets."""
    print("COMPREHENSIVE DATA PREPROCESSING")
    print("="*80)
    print("Following the 7-step preprocessing guidelines:")
    print("1. Scaling (z-transformation)")
    print("2. Centering (mean=0 for PCA)")
    print("3. Categorical variables (one-hot encoding)")
    print("4. Missing values (none found)")
    print("5. Outlier treatment (robust scaling)")
    print("6. Dataset splitting (train/test)")
    print("7. Target variable preparation")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(outlier_method='robust', test_size=0.2, random_state=42)
    
    # Define dataset configurations
    dataset_configs = {
        'california_housing': {
            'file': 'california_housing/california_housing.csv',
            'target': 'target',
            'categorical': []
        },
        'diabetes': {
            'file': 'diabetes/diabetes.csv',
            'target': 'target',
            'categorical': []
        },
        'energy_efficiency': {
            'file': 'energy_efficiency/energy_efficiency.csv',
            'target': 'Y2',  # Cooling load
            'categorical': []
        },
        'concrete_strength': {
            'file': 'concrete_strength/concrete_strength.csv',
            'target': 'Concrete compressive strength(MPa, megapascals) ',
            'categorical': []
        },
        'wine_quality_red': {
            'file': 'wine_quality/wine_quality_red.csv',
            'target': 'quality',
            'categorical': ['wine_type']
        },
        'wine_quality_white': {
            'file': 'wine_quality/wine_quality_white.csv',
            'target': 'quality',
            'categorical': ['wine_type']
        },
        'wine_quality_combined': {
            'file': 'wine_quality/wine_quality_combined.csv',
            'target': 'quality',
            'categorical': ['wine_type']
        },
        'airfoil_noise': {
            'file': 'airfoil_noise/airfoil_self_noise.csv',
            'target': 'Scaled_sound_pressure_level',
            'categorical': []
        },
        'bike_sharing_day': {
            'file': 'bike_sharing/day.csv',
            'target': 'cnt',
            'categorical': ['dteday']
        },
        'bike_sharing_hour': {
            'file': 'bike_sharing/hour.csv',
            'target': 'cnt',
            'categorical': ['dteday']
        },
        'abalone': {
            'file': 'abalone/abalone.csv',
            'target': 'Rings',
            'categorical': ['Sex']
        }
    }
    
    # Process all datasets
    processed_datasets = {}
    
    for name, config in dataset_configs.items():
        if os.path.exists(config['file']):
            try:
                processed_data = preprocessor.process_dataset(
                    config['file'], 
                    name, 
                    config['target'], 
                    config['categorical']
                )
                processed_datasets[name] = processed_data
                
                # Save processed dataset
                output_dir = f"processed/{name}"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Save as CSV files
                processed_data['X_train_scaled'].to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
                processed_data['X_test_scaled'].to_csv(f"{output_dir}/X_test_scaled.csv", index=False)
                processed_data['y_train'].to_csv(f"{output_dir}/y_train.csv", index=False, header=['target'])
                processed_data['y_test'].to_csv(f"{output_dir}/y_test.csv", index=False, header=['target'])
                
                # Save unscaled versions for comparison
                processed_data['X_train'].to_csv(f"{output_dir}/X_train_raw.csv", index=False)
                processed_data['X_test'].to_csv(f"{output_dir}/X_test_raw.csv", index=False)
                
                # Save metadata
                metadata = {
                    'dataset_name': name,
                    'target_name': processed_data['target_name'],
                    'feature_names': processed_data['feature_names'],
                    'original_shape': processed_data['original_shape'],
                    'processed_shape': processed_data['processed_shape'],
                    'preprocessing_steps': [
                        'StandardScaler/RobustScaler applied',
                        'Data centered (mean ≈ 0)',
                        'Categorical variables encoded',
                        'Train/test split (80/20)',
                        'Outliers handled with robust scaling'
                    ]
                }
                
                import json
                with open(f"{output_dir}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✓ Saved processed data to: {output_dir}/")
                
            except Exception as e:
                print(f"✗ Error processing {name}: {e}")
        else:
            print(f"✗ File not found: {config['file']}")
    
    # Create summary report
    print(f"\n{'='*80}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nSuccessfully processed {len(processed_datasets)} datasets:")
    for name, data in processed_datasets.items():
        print(f"  {name}:")
        print(f"    Original: {data['original_shape'][0]} × {data['original_shape'][1]}")
        print(f"    Processed: {data['processed_shape'][0]} × {data['processed_shape'][1]} (train)")
        print(f"    Features: {len(data['feature_names'])}")
        print(f"    Target: {data['target_name']}")
    
    print(f"\nAll datasets are now ready for:")
    print(f"  ✓ Principal Component Analysis (PCA)")
    print(f"  ✓ Linear Regression")
    print(f"  ✓ Comparative analysis of dimensionality reduction methods")
    
    print(f"\nProcessed datasets saved to: ./processed/ directory")
    print(f"Each dataset includes:")
    print(f"  - X_train_scaled.csv, X_test_scaled.csv (standardized features)")
    print(f"  - y_train.csv, y_test.csv (target variables)")
    print(f"  - X_train_raw.csv, X_test_raw.csv (unscaled for reference)")
    print(f"  - metadata.json (preprocessing information)")

if __name__ == "__main__":
    main()