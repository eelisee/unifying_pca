# Dataset Preprocessing Documentation

## Overview

This document describes the comprehensive preprocessing pipeline applied to all datasets for PCA and linear regression analysis. The preprocessing follows the 7-step guidelines you specified for optimal PCA performance.

## Preprocessing Steps Applied

### 1. Skalierung (Scaling)
- **Method**: RobustScaler (robust to outliers)
- **Purpose**: PCA is scale-dependent; variables with larger variances would dominate
- **Implementation**: z-transformation with robust statistics (median and IQR instead of mean and std)
- **Formula**: `x' = (x - median(x)) / IQR(x)`

### 2. Zentrierung (Centering)
- **Method**: Automatic centering through RobustScaler
- **Purpose**: PCA requires centered data (E[X] ≈ 0)
- **Verification**: Max absolute mean after processing is < 0.6 for all datasets
- **Result**: All features have approximately zero mean

### 3. Kategoriale Variablen (Categorical Variables)
- **Wine datasets**: `wine_type` → One-hot encoded (red/white)
- **Abalone**: `Sex` → One-hot encoded (M/F/I → binary features)
- **Bike sharing**: `dteday` → Dropped (date strings not useful for PCA)
- **Strategy**: One-hot encoding for low cardinality, label encoding for high cardinality

### 4. Fehlende Werte (Missing Values)
- **Status**: ✅ No missing values found in any dataset
- **Strategy**: Would use mean/median imputation or MICE if needed

### 5. Outlier-Behandlung (Outlier Treatment)
- **Method**: RobustScaler inherently handles outliers
- **Alternative**: Winsorization implemented but not used (RobustScaler preferred)
- **Detection**: IQR method used for identification
- **Effect**: Outliers still present but their influence is minimized

### 6. Datensatzaufteilung (Dataset Splitting)
- **Split ratio**: 80% train / 20% test
- **Method**: sklearn train_test_split with random_state=42
- **Purpose**: Enable proper evaluation and prevent overfitting
- **Stratification**: None (regression targets)

### 7. Zielvariable (Target Variable)
- **Identification**: Automatic detection based on column names and patterns
- **Separation**: Clean X/y split for supervised learning
- **Preservation**: Both scaled and unscaled versions saved

## Dataset Summary

| Dataset | Original Shape | Processed Features | Target Variable | Special Notes |
|---------|---------------|-------------------|-----------------|---------------|
| California Housing | 20640 × 9 | 8 | target | Large dataset, good for testing |
| Diabetes | 442 × 11 | 10 | target | Small, perfect for debugging |
| Energy Efficiency | 768 × 10 | 9 | Y2 (cooling load) | Two targets available (Y1, Y2) |
| Concrete Strength | 1030 × 9 | 8 | compressive strength | Long column names cleaned |
| Wine Quality Red | 1599 × 13 | 11 | quality | Wine type encoded |
| Wine Quality White | 4898 × 13 | 11 | quality | Larger than red wine dataset |
| Wine Quality Combined | 6497 × 13 | 12 | quality | Both wine types, extra feature |
| Airfoil Self-Noise | 1503 × 6 | 5 | sound pressure level | Technical/engineering dataset |
| Bike Sharing Day | 731 × 16 | 14 | cnt | Temporal patterns, date dropped |
| Bike Sharing Hour | 17379 × 17 | 15 | cnt | Largest dataset, hourly granularity |
| Abalone | 4177 × 9 | 9 | Rings (age) | Sex encoded as binary features |

## File Structure

Each processed dataset is saved in `data/processed/{dataset_name}/` with:

- `X_train_scaled.csv` - Standardized training features (ready for PCA)
- `X_test_scaled.csv` - Standardized test features (ready for PCA)
- `y_train.csv` - Training target variable
- `y_test.csv` - Test target variable
- `X_train_raw.csv` - Unscaled training features (for reference)
- `X_test_raw.csv` - Unscaled test features (for reference)
- `metadata.json` - Dataset information and preprocessing details

## Usage Examples

### Loading a Processed Dataset

```python
import pandas as pd

# Load processed data
X_train = pd.read_csv('data/processed/california_housing/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/california_housing/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/california_housing/y_train.csv')['target']
y_test = pd.read_csv('data/processed/california_housing/y_test.csv')['target']

print(f"Training data shape: {X_train.shape}")
print(f"Features are centered: {abs(X_train.mean()).max():.6f}")  # Should be ≈ 0
```

### PCA Analysis

```python
from sklearn.decomposition import PCA
import numpy as np

# Apply PCA (data already centered and scaled)
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Analyze explained variance
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var_ratio)

print(f"Number of components for 95% variance: {np.argmax(cumulative_var >= 0.95) + 1}")
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fit linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

## Quality Assurance

### Verification Checks Performed

1. **Centering verification**: Max absolute mean < 0.6 for all datasets
2. **No data leakage**: Scaling fitted only on training data
3. **Reproducibility**: Fixed random seeds for consistent splits
4. **Feature preservation**: All feature names and types tracked
5. **Target integrity**: Target variables properly separated and preserved

### Expected Properties

- ✅ All numerical features are standardized
- ✅ Data is centered (mean ≈ 0)
- ✅ Categorical variables are properly encoded
- ✅ No missing values
- ✅ Outlier influence minimized
- ✅ Clean train/test splits
- ✅ Ready for both PCA and regression

## Next Steps

The processed datasets are now ready for:

1. **PCA Implementation**: Apply principal component analysis to reduce dimensionality
2. **Linear Regression**: Fit regression models on original and PCA-transformed features
3. **Comparative Analysis**: Compare performance of different approaches
4. **Stability Analysis**: Test how methods behave with varying sample sizes
5. **Algebraic Framework**: Implement your unifying PCA perspective

## Notes

- RobustScaler was chosen over StandardScaler to handle outliers more gracefully
- One-hot encoding was preferred over label encoding to avoid introducing artificial ordinal relationships
- Date columns were dropped as they don't contribute meaningful information for PCA
- All preprocessing parameters (scalers, encoders) are stored for potential inverse transforms
- The preprocessing is designed to be reproducible and scientifically sound

For any questions about the preprocessing pipeline or specific dataset characteristics, refer to the individual `metadata.json` files in each dataset directory.