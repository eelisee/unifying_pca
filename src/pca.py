"""
Standard PCA implementation within the algebraic operator framework.

This module implements PCA as projection operators in class P, where A_σ = 0
and A_μ is a low-rank projection matrix.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base import OperatorP, validate_data, DataMetrics


class PCAOperator(OperatorP):
    """
    PCA as a projection operator in class P.
    
    In the algebraic framework, PCA corresponds to operators A ∈ P where:
    - A_σ = 0 (no explicit noise term)
    - A_β can be set to reconstruction coefficients or zeros
    - A_μ = Q_r Q_r^T is a projection matrix (rank r)
    
    This implements both standard PCA and its use for regression via reconstruction.
    """
    
    def __init__(self, k: int, n_components: Optional[int] = None, 
                 center: bool = True, scale: bool = False):
        """
        Initialize PCA operator.
        
        Parameters:
        k: Dimension of augmented space
        n_components: Number of principal components to keep
        center: Whether to center the data
        scale: Whether to scale the data to unit variance
        """
        super().__init__(k)
        self.n_components = n_components if n_components is not None else self.p
        self.center = center
        self.scale = scale
        
        # PCA has no explicit noise term
        self.A_sigma = 0.0
        
        # Will be set during fitting
        self.components_ = None  # Q_r matrix
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.scale_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PCAOperator':
        """
        Fit PCA operator to data.
        
        Parameters:
        X: Feature matrix of shape (n, p)
        y: Ignored (PCA is unsupervised)
        
        Returns:
        Self (fitted operator)
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        n, p = X.shape
        
        # Validate n_components
        self.n_components = min(self.n_components, min(n, p))
        
        # Preprocessing
        X_processed = X.copy()
        
        # Centering
        if self.center:
            self.mean_ = np.mean(X_processed, axis=0)
            X_processed = X_processed - self.mean_
        else:
            self.mean_ = np.zeros(p)
        
        # Scaling
        if self.scale:
            self.scale_ = np.std(X_processed, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
            X_processed = X_processed / self.scale_
        else:
            self.scale_ = np.ones(p)
        
        # Compute SVD
        U, s, Vt = np.linalg.svd(X_processed, full_matrices=False)
        
        # Extract components (principal directions)
        self.components_ = Vt[:self.n_components]  # Q_r matrix (r × p)
        
        # Explained variance
        explained_variance = (s ** 2) / (n - 1) if n > 1 else s ** 2
        self.explained_variance_ = explained_variance[:self.n_components]
        total_variance = np.sum(explained_variance)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance if total_variance > 0 else 
            np.zeros(self.n_components)
        )
        
        # Set A_μ as projection matrix Q_r Q_r^T
        self.A_mu = self.components_.T @ self.components_
        
        # For reconstruction-based regression, A_β can be set
        # Here we initialize to zeros (pure PCA)
        self.A_beta = np.zeros(self.p)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Parameters:
        X: Feature matrix of shape (n, p)
        
        Returns:
        Transformed data of shape (n, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transforming")
        
        X = np.asarray(X)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        # Apply same preprocessing as during fit
        X_processed = X - self.mean_
        if self.scale:
            X_processed = X_processed / self.scale_
        
        # Project onto principal components
        return X_processed @ self.components_.T
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back from principal component space.
        
        This implements the reconstruction X_reconstructed = A_μ X.
        
        Parameters:
        X_transformed: Data in PC space of shape (n, n_components)
        
        Returns:
        Reconstructed data of shape (n, p)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse transforming")
        
        X_transformed = np.asarray(X_transformed)
        if X_transformed.shape[1] != self.n_components:
            raise ValueError(f"Expected {self.n_components} components, got {X_transformed.shape[1]}")
        
        # Reconstruct in original space
        X_reconstructed = X_transformed @ self.components_
        
        # Reverse preprocessing
        if self.scale:
            X_reconstructed = X_reconstructed * self.scale_
        X_reconstructed = X_reconstructed + self.mean_
        
        return X_reconstructed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit PCA and transform data in one step."""
        return self.fit(X, y).transform(X)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data using PCA projection.
        
        This corresponds to applying the operator A_μ = Q_r Q_r^T.
        """
        X_transformed = self.transform(X)
        return self.inverse_transform(X_transformed)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For pure PCA, prediction is reconstruction.
        
        This can be overridden in subclasses for regression-based PCA.
        """
        return self.reconstruct(X)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample.
        
        Returns:
        Per-sample reconstruction errors of shape (n,)
        """
        X_reconstructed = self.reconstruct(X)
        return np.sum((X - X_reconstructed) ** 2, axis=1)
    
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute PCA performance metrics.
        
        For pure PCA, we measure reconstruction quality.
        """
        X_reconstructed = self.reconstruct(X)
        
        # Reconstruction metrics
        mse_reconstruction = np.mean((X - X_reconstructed) ** 2)
        explained_var_ratio = DataMetrics.explained_variance_ratio(X, X_reconstructed)
        
        metrics = {
            "reconstruction_mse": mse_reconstruction,
            "explained_variance_ratio": explained_var_ratio,
            "cumulative_explained_variance": np.sum(self.explained_variance_ratio_),
        }
        
        # If y is provided, compute prediction metrics
        if y is not None:
            y_pred = self.predict(X)
            if y_pred.shape == y.shape:
                metrics.update({
                    "prediction_mse": DataMetrics.mse(y, y_pred),
                    "prediction_r2": DataMetrics.r2_score(y, y_pred)
                })
        
        return metrics
    
    def get_operator_info(self) -> Dict[str, Any]:
        """Get information about the fitted PCA operator."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "type": "PCA",
            "A_sigma": self.A_sigma,
            "A_beta": self.A_beta.copy(),
            "A_mu": self.A_mu.copy(),
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio_.copy(),
            "cumulative_explained_variance": np.sum(self.explained_variance_ratio_),
            "components": self.components_.copy(),
            "center": self.center,
            "scale": self.scale
        }


class PCARegressionOperator(PCAOperator):
    """
    PCA-based regression operator.
    
    This implements regression in the reduced PC space, corresponding to
    operators where A_μ is a projection and A_β represents regression
    coefficients in the principal component space.
    """
    
    def __init__(self, k: int, n_components: Optional[int] = None, 
                 center: bool = True, scale: bool = False, 
                 regression_regularization: Optional[str] = None, alpha: float = 1.0):
        """
        Initialize PCA regression operator.
        
        Parameters:
        k: Dimension of augmented space
        n_components: Number of principal components for regression
        center: Whether to center the data
        scale: Whether to scale the data
        regression_regularization: Regularization for regression step
        alpha: Regularization strength
        """
        super().__init__(k, n_components, center, scale)
        self.regression_regularization = regression_regularization
        self.alpha = alpha
        self.regression_coefficients_ = None
        self.intercept_ = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PCARegressionOperator':
        """
        Fit PCA followed by regression in PC space.
        
        Parameters:
        X: Feature matrix of shape (n, p)
        y: Target vector of shape (n,)
        
        Returns:
        Self (fitted operator)
        """
        X, y = validate_data(X, y)
        
        # First fit PCA
        super().fit(X, y)
        
        # Transform X to PC space
        X_pc = self.transform(X)
        
        # Fit regression in PC space
        if self.regression_regularization == 'ridge':
            self._fit_ridge_regression(X_pc, y)
        else:
            self._fit_ols_regression(X_pc, y)
        
        # Set A_β in original space
        # A_β = regression_coefficients @ Q_r
        self.A_beta = self.regression_coefficients_ @ self.components_
        
        return self
    
    def _fit_ols_regression(self, X_pc: np.ndarray, y: np.ndarray) -> None:
        """Fit OLS regression in PC space."""
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X_pc.shape[0]), X_pc])
        
        # Solve normal equations
        try:
            coefficients = np.linalg.solve(X_with_intercept.T @ X_with_intercept, 
                                         X_with_intercept.T @ y)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(X_with_intercept) @ y
        
        self.intercept_ = coefficients[0]
        self.regression_coefficients_ = coefficients[1:]
    
    def _fit_ridge_regression(self, X_pc: np.ndarray, y: np.ndarray) -> None:
        """Fit ridge regression in PC space."""
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X_pc.shape[0]), X_pc])
        n_features = X_with_intercept.shape[1]
        
        # Regularization matrix (don't regularize intercept)
        I = np.eye(n_features)
        I[0, 0] = 0
        
        coefficients = np.linalg.solve(
            X_with_intercept.T @ X_with_intercept + self.alpha * I,
            X_with_intercept.T @ y
        )
        
        self.intercept_ = coefficients[0]
        self.regression_coefficients_ = coefficients[1:]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using PCA + regression.
        
        The prediction follows: y = A_β^T X + intercept
        where A_β = regression_coefficients @ Q_r
        """
        if not self.is_fitted or self.regression_coefficients_ is None:
            raise ValueError("PCARegressionOperator must be fitted before predicting")
        
        # Method 1: Transform to PC space and apply regression
        X_pc = self.transform(X)
        y_pred = X_pc @ self.regression_coefficients_ + self.intercept_
        
        return y_pred
    
    def get_operator_info(self) -> Dict[str, Any]:
        """Get information about the fitted PCA regression operator."""
        info = super().get_operator_info()
        if self.is_fitted and self.regression_coefficients_ is not None:
            info.update({
                "type": "PCARregression",
                "regression_coefficients": self.regression_coefficients_.copy(),
                "intercept": self.intercept_,
                "regression_regularization": self.regression_regularization,
                "alpha": self.alpha if self.regression_regularization else None
            })
        return info


class PCAFamily:
    """
    Family of PCA operators corresponding to different rank constraints.
    
    This implements the H_r operator families described in the theory,
    where H_r consists of operators with rank at most r.
    """
    
    def __init__(self, k: int, max_components: Optional[int] = None, **kwargs):
        """
        Initialize family of PCA operators.
        
        Parameters:
        k: Dimension of augmented space
        max_components: Maximum number of components to consider
        **kwargs: Additional parameters for PCA operators
        """
        self.k = k
        self.p = k - 1
        self.max_components = max_components if max_components is not None else self.p
        self.kwargs = kwargs
        self.operators = {}
    
    def fit_H_r(self, X: np.ndarray, y: Optional[np.ndarray] = None, r: int = 1, 
                regression: bool = False) -> OperatorP:
        """
        Fit operator from H_r (rank-r projection operators).
        
        Parameters:
        X: Feature matrix
        y: Target vector (required if regression=True)
        r: Rank constraint
        regression: Whether to fit PCA regression or pure PCA
        
        Returns:
        Fitted PCA operator with rank r
        """
        if not (1 <= r <= min(self.max_components, X.shape[0], X.shape[1])):
            raise ValueError(f"r must be between 1 and {min(self.max_components, X.shape[0], X.shape[1])}")
        
        if regression:
            if y is None:
                raise ValueError("y must be provided for PCA regression")
            operator = PCARegressionOperator(self.k, n_components=r, **self.kwargs)
            operator.fit(X, y)
        else:
            operator = PCAOperator(self.k, n_components=r, **self.kwargs)
            operator.fit(X, y)
        
        # Store in family
        key = f"H_{r}_regression" if regression else f"H_{r}"
        self.operators[key] = operator
        return operator
    
    def fit_all_ranks(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                     regression: bool = False) -> Dict[str, OperatorP]:
        """Fit PCA operators for all ranks from 1 to max_components."""
        max_rank = min(self.max_components, X.shape[0], X.shape[1])
        
        for r in range(1, max_rank + 1):
            self.fit_H_r(X, y, r, regression)
        
        suffix = "_regression" if regression else ""
        return {k: v for k, v in self.operators.items() if k.endswith(suffix)}
    
    def compare_ranks(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of PCA operators with different ranks.
        
        Returns:
        Dictionary mapping operator names to their performance metrics
        """
        results = {}
        
        for name, operator in self.operators.items():
            results[name] = operator.score(X_test, y_test)
        
        return results
    
    def analyze_stability(self, X1: np.ndarray, X2: np.ndarray, 
                         y1: Optional[np.ndarray] = None, y2: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze operator stability ||H^(n) - H^(n+m)||_F between two datasets.
        
        This measures extendability as described in the theory.
        """
        stability_results = {}
        
        # Extract rank numbers from operator names and sort
        rank_operators = {}
        for name, operator in self.operators.items():
            if name.startswith("H_"):
                parts = name.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    rank = int(parts[1])
                    is_regression = "regression" in name
                    rank_operators[(rank, is_regression)] = operator
        
        for (rank, is_regression), operator1 in rank_operators.items():
            # Fit same type of operator on second dataset
            if is_regression:
                if y2 is None:
                    continue
                operator2 = PCARegressionOperator(self.k, n_components=rank, **self.kwargs)
                operator2.fit(X2, y2)
            else:
                operator2 = PCAOperator(self.k, n_components=rank, **self.kwargs)
                operator2.fit(X2)
            
            # Compare A_μ matrices (projection operators)
            H1 = operator1.A_mu
            H2 = operator2.A_mu
            
            stability = DataMetrics.operator_stability(H1, H2, norm='fro')
            
            suffix = "_regression" if is_regression else ""
            stability_results[f"H_{rank}{suffix}"] = stability
        
        return stability_results