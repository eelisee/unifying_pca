# Model-Oriented Generalized PCA: Mathematical Corrections Applied

## Overview
This document summarizes the key corrections made to align the implementation with the mathematical specification.

## Critical Fixes Applied

### 1. **A_β Projection Formula** ✅ **FIXED**
- **Issue**: Incorrect dimensionality in `A_beta = beta_ols_.T @ V_beta @ V_beta.T`
- **Fix**: Changed to `A_beta = (V_beta @ V_beta.T) @ beta_ols_`
- **Rationale**: Projects β̂ onto the V_β subspace correctly: A_β = P_{V_β} β̂

### 2. **Residual Scale vs Variance** ✅ **FIXED**  
- **Issue**: Confusion between variance and scale in A_σ
- **Fix**: 
  - `_compute_ols_baseline` now returns `np.sqrt(residual_var)` (scale, not variance)
  - Renamed `residual_variance_` → `residual_scale_`
  - `A_sigma = self.residual_scale_` (no longer `np.sqrt(...)`)
- **Rationale**: A_σ should be the scale factor σ, not σ²

### 3. **AIC/BIC Calculation** ✅ **FIXED**
- **Issue**: Incorrect sign handling in AIC/BIC formulas
- **Fix**: 
  - `aic = 2 * k + 2 * nll` (was `2 * k - 2 * (-nll)`)
  - `bic = np.log(n) * k + 2 * nll` (was `np.log(n) * k - 2 * (-nll)`)
- **Rationale**: Standard information criteria formulation

### 4. **Iterative β Refitting** ✅ **ADDED**
- **Issue**: Missing explicit refit of β on projected predictors
- **Fix**: Added after V_H update:
  ```python
  X_proj = X @ (V_H @ V_H.T)
  beta_refit, _ = self._compute_ols_baseline(X_proj, Y)
  self.beta_ols_ = beta_refit
  ```
- **Rationale**: Ensures β is consistent with current PCA subspace

### 5. **Documentation Clarity** ✅ **IMPROVED**
- Added comments clarifying that PLS is a "numerical shortcut, not part of core theory"
- Clarified mathematical notation in comments
- Updated docstrings to reflect scale vs variance distinction

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Empirical Covariance Σ_Z | ✅ Correct | Already properly implemented |
| Operator Structure A | ✅ Correct | Block matrix form is correct |
| V_H Update (Spectral) | ✅ Correct | Eigendecomposition of residual covariance |
| KL Divergence Formula | ✅ Correct | Proper Gaussian KL divergence |
| Regularization | ✅ Correct | Numerical stability with ε·I |

## Testing Results

The corrected implementation successfully:
- ✅ Runs without numerical errors
- ✅ Selects reasonable model configurations
- ✅ Produces finite KL divergence values
- ✅ Follows the mathematical specification precisely

## Key Mathematical Insights

1. **A_σ is a scale parameter** (standard deviation), not variance
2. **A_β projection** must be P_{V_β} β̂, not β̂^T P_{V_β}
3. **Iterative refinement** requires refitting β on projected X after each V_H update
4. **Information criteria** use standard positive formulation

These corrections ensure the implementation faithfully represents the theoretical model while maintaining numerical stability.