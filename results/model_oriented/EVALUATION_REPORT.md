# Model-Oriented Generalized PCA Evaluation Report

Generated on: 2025-10-07 23:12:28

## Overview

This report presents the evaluation results of the Model-Oriented Generalized PCA implementation across multiple datasets, comparing it against baseline methods.

## Methodology

The Model-Oriented Generalized PCA uses:
- Alternating subspace optimization
- KL divergence minimization between empirical and model-induced distributions
- Joint optimization over regression subspace (V_β) and PCA subspace (V_H)
- Model selection via BIC criterion

## Datasets Evaluated

- bike_sharing_day
- wine_quality_red
- airfoil_noise
- abalone
- bike_sharing_hour
- wine_quality_combined
- diabetes
- wine_quality_white
- energy_efficiency
- concrete_strength
- california_housing

## Models Compared

1. **ModelOrientedPCA**: Our new implementation
2. **GeneralizedPCA**: Previous generalized PCA implementation
3. **PCA+LinearReg**: Standard PCA followed by linear regression
4. **LinearRegression**: Pure linear regression baseline
5. **PLSRegression**: Partial Least Squares regression

## Performance Summary

### Average Performance Across All Datasets

| Model | Avg Test R² | Std Test R² | Avg Test MSE | Avg Fit Time (s) |
|-------|-------------|-------------|--------------|------------------|
| PLSRegression | -130.3203 | 434.1139 | 12407.4532 | 0.0028 |

## Key Findings

Please refer to the detailed results and visualizations for comprehensive analysis.

## Files Generated

- `detailed_results.json`: Complete evaluation results
- `evaluation_summary.csv`: Summary statistics
- `performance_summary.csv`: Performance comparison by model
- `evaluation_comparison.png`: Overall comparison visualizations
- `[dataset]_detailed_comparison.png`: Per-dataset detailed plots

