# Datasets for PCA Research

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
