#!/usr/bin/env python3
"""
Script to download and organize datasets for PCA research.
Downloads datasets from sklearn, UCI ML Repository, and other sources.
"""

import os
import pandas as pd
import numpy as np
from sklearn import datasets
import urllib.request
import zipfile
import io
import requests
from pathlib import Path
import ssl
import warnings

# Disable SSL verification for downloads (only for this script)
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'boston_housing', 'california_housing', 'diabetes', 'energy_efficiency',
        'concrete_strength', 'wine_quality', 'airfoil_noise', 'bike_sharing', 'abalone'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Created all necessary directories")

def download_sklearn_datasets():
    """Download and save sklearn built-in datasets."""
    print("Downloading sklearn datasets...")
    
    # Boston Housing (deprecated but still available)
    try:
        boston = datasets.load_boston()
        boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
        boston_df['target'] = boston.target
        boston_df.to_csv('boston_housing/boston_housing.csv', index=False)
        print("✓ Boston Housing dataset saved")
    except ImportError:
        print("⚠ Boston Housing dataset not available (deprecated in newer sklearn versions)")
    
    # California Housing
    california = datasets.fetch_california_housing()
    california_df = pd.DataFrame(california.data, columns=california.feature_names)
    california_df['target'] = california.target
    california_df.to_csv('california_housing/california_housing.csv', index=False)
    print("✓ California Housing dataset saved")
    
    # Diabetes
    diabetes = datasets.load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    diabetes_df.to_csv('diabetes/diabetes.csv', index=False)
    print("✓ Diabetes dataset saved")

def download_uci_datasets():
    """Download UCI datasets."""
    print("Downloading UCI datasets...")
    
    # Energy Efficiency Dataset
    try:
        energy_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        energy_df = pd.read_excel(energy_url)
        energy_df.to_csv('energy_efficiency/energy_efficiency.csv', index=False)
        print("✓ Energy Efficiency dataset saved")
    except Exception as e:
        print(f"⚠ Energy Efficiency dataset failed: {e}")
    
    # Concrete Compressive Strength
    try:
        concrete_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        concrete_df = pd.read_excel(concrete_url)
        concrete_df.to_csv('concrete_strength/concrete_strength.csv', index=False)
        print("✓ Concrete Compressive Strength dataset saved")
    except Exception as e:
        print(f"⚠ Concrete dataset failed: {e}")
    
    # Wine Quality (Red and White)
    try:
        # Red wine
        red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        red_wine_df = pd.read_csv(red_wine_url, sep=';')
        red_wine_df['wine_type'] = 'red'
        
        # White wine
        white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        white_wine_df = pd.read_csv(white_wine_url, sep=';')
        white_wine_df['wine_type'] = 'white'
        
        # Save separately and combined
        red_wine_df.to_csv('wine_quality/wine_quality_red.csv', index=False)
        white_wine_df.to_csv('wine_quality/wine_quality_white.csv', index=False)
        
        combined_wine = pd.concat([red_wine_df, white_wine_df], ignore_index=True)
        combined_wine.to_csv('wine_quality/wine_quality_combined.csv', index=False)
        print("✓ Wine Quality datasets saved (red, white, and combined)")
    except Exception as e:
        print(f"⚠ Wine Quality datasets failed: {e}")
    
    # Airfoil Self-Noise
    try:
        airfoil_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
        response = requests.get(airfoil_url)
        airfoil_data = io.StringIO(response.text)
        airfoil_df = pd.read_csv(airfoil_data, sep='\t', header=None)
        airfoil_df.columns = ['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement', 'Scaled_sound_pressure_level']
        airfoil_df.to_csv('airfoil_noise/airfoil_self_noise.csv', index=False)
        print("✓ Airfoil Self-Noise dataset saved")
    except Exception as e:
        print(f"⚠ Airfoil dataset failed: {e}")
    
    # Bike Sharing Dataset
    try:
        bike_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
        response = requests.get(bike_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall('bike_sharing/')
        print("✓ Bike Sharing dataset saved")
    except Exception as e:
        print(f"⚠ Bike Sharing dataset failed: {e}")
    
    # Abalone Dataset
    try:
        abalone_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        abalone_df = pd.read_csv(abalone_url, header=None)
        abalone_df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
        abalone_df.to_csv('abalone/abalone.csv', index=False)
        print("✓ Abalone dataset saved")
    except Exception as e:
        print(f"⚠ Abalone dataset failed: {e}")

def create_readme():
    """Create a README file describing all datasets."""
    readme_content = """# Datasets for PCA Research

This directory contains various datasets collected for Principal Component Analysis research.

## Dataset Overview

### 1. Boston Housing (boston_housing/)
- **Source**: sklearn/UCI
- **Description**: House price prediction based on socioeconomic variables
- **Features**: Strongly correlated variables - ideal for PCA comparison
- **Note**: Deprecated in newer sklearn versions due to ethical concerns

### 2. California Housing (california_housing/)
- **Source**: sklearn/OpenML
- **Description**: Updated version of Boston dataset, robust and large (20k observations)
- **Features**: Excellent PCA/regression test base

### 3. Diabetes Dataset (diabetes/)
- **Source**: sklearn
- **Description**: Disease progression prediction from 10 numerical variables
- **Features**: Small but perfect for debugging

### 4. Energy Efficiency Dataset (energy_efficiency/)
- **Source**: UCI
- **Description**: Heating/cooling demand prediction from physical building variables
- **Features**: Contains nonlinearities and correlations

### 5. Concrete Compressive Strength (concrete_strength/)
- **Source**: UCI
- **Description**: Concrete strength from material proportions
- **Features**: Realistic, moderate dimension, strongly correlated

### 6. Wine Quality (wine_quality/)
- **Source**: UCI/Kaggle
- **Description**: Wine quality from chemical variables
- **Features**: Moderate dimension, numerical, easily interpretable

### 7. Airfoil Self-Noise (airfoil_noise/)
- **Source**: UCI
- **Description**: Aerodynamic simulation, sound level prediction
- **Features**: Strongly correlated technical variables

### 8. Bike Sharing Dataset (bike_sharing/)
- **Source**: UCI/Kaggle
- **Description**: Bicycle demand over time, weather, calendar data
- **Features**: Temporal patterns, multiple predictors, clear target

### 9. Abalone Dataset (abalone/)
- **Source**: UCI
- **Description**: Abalone age from measurements
- **Features**: Classic dataset, good linear/nonlinear mix

## Usage

Each dataset is stored in its own subdirectory with appropriate CSV files.
Use these datasets to compare different PCA implementations and methods.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("✓ README.md created")

def main():
    """Main function to download all datasets."""
    print("Starting dataset download process...")
    print("=" * 50)
    
    create_directories()
    print()
    
    download_sklearn_datasets()
    print()
    
    download_uci_datasets()
    print()
    
    create_readme()
    print()
    
    print("=" * 50)
    print("Dataset download process completed!")
    print("Check individual directories for downloaded files.")

if __name__ == "__main__":
    main()