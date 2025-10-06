"""
Generalized PCA implementation based on the algebraic framework from Chapter 4.

This module implements the operator class P from the paper, treating PCA as 
an operator-choice problem in the block matrix structure:

A = [A_σ    A_β  ]
    [0      A_μ  ]

where the generalized PCA optimizes the choice of operator H to minimize
the model-theoretic loss L(μ, Hμ).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from .base import OperatorP, validate_data, DataMetrics
    from .pca import PCAOperator
except ImportError:
    # Fallback for direct execution
    from base import OperatorP, validate_data, DataMetrics
    from pca import PCAOperator


class GeneralizedPCAOperator(OperatorP):
    """
    Generalized PCA operator implementing the operator class P from Chapter 4.
    
    This implements the operator-choice problem (Eq. 4.3):
    PCA_r(μ) = argmin_{H ∈ H, rank(H) ≤ r} L(μ, Hμ)
    
    The implementation follows the block matrix structure and uses empirical 
    approximation of the model-theoretic loss.
    """
    
    def __init__(self, k: int, n_components: Optional[int] = None,
                 use_empirical_loss: bool = True, center: bool = True,
                 loss_type: Optional[str] = None, optimization_method: Optional[str] = None,
                 max_iter: Optional[int] = None, scale: Optional[bool] = None,
                 **kwargs):
        """
        Initialize Generalized PCA operator.
        
        Parameters:
        k: Dimension of augmented space (k = p + 1 where p is number of predictors)
        n_components: Number of components to extract (rank constraint)
        use_empirical_loss: Whether to use empirical MSE approximation of model loss
        center: Whether to center the data
        loss_type: Legacy parameter (ignored, for backwards compatibility)
        optimization_method: Legacy parameter (ignored, for backwards compatibility)
        max_iter: Legacy parameter (ignored, for backwards compatibility)  
        scale: Legacy parameter (ignored, for backwards compatibility)
        **kwargs: Additional legacy parameters (ignored)
        """
        super().__init__(k)
        self.n_components = n_components if n_components is not None else self.p
        self.use_empirical_loss = use_empirical_loss
        self.center = center
        
        # Legacy parameters for backwards compatibility (not used in simplified implementation)
        self.loss_type = loss_type
        self.optimization_method = optimization_method
        self.max_iter = max_iter
        self.scale = scale
        
        # Initialize operator components according to Chapter 4
        self.A_sigma = 0.0  # No noise term in PCA
        self.A_beta = np.zeros(self.p)  # No direct regression coefficients in pure PCA
        # A_mu will be set during fitting as the projection operator
        
        # Storage for fitted components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'GeneralizedPCAOperator':
        """
        Fit generalized PCA operator using the operator-choice framework.
        
        This implements the operator-choice problem by finding the optimal 
        low-rank operator H that minimizes the empirical approximation of L(μ, Hμ).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        n, p = X.shape
        self.n_components = min(self.n_components, min(n, p))
        
        # Preprocessing
        if self.center:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
        else:
            self.mean_ = np.zeros(p)
            X_centered = X.copy()
        
        # Solve operator-choice problem: find optimal H
        # Following Chapter 4, this is implemented as finding the projection operator
        # that minimizes the empirical loss
        H_optimal = self._solve_operator_choice_problem(X_centered)
        
        # Set the A_mu component of the operator
        self.A_mu = H_optimal
        
        # Extract principal components from the optimal operator
        self._extract_components_from_operator(H_optimal, X_centered)
        
        self.is_fitted = True
        return self
    
    def _solve_operator_choice_problem(self, X: np.ndarray) -> np.ndarray:
        """
        Solve the operator-choice problem from Eq. 4.3 in Chapter 4.
        
        This finds H that minimizes L(μ, Hμ) subject to rank(H) ≤ r.
        For empirical implementation, we use the data-based approximation.
        """
        n, p = X.shape
        
        if self.use_empirical_loss:
            # Empirical approximation: Use SVD-based approach
            # This corresponds to minimizing the empirical reconstruction error
            # as an approximation of the model-theoretic loss L(μ, Hμ)
            
            # Compute covariance matrix
            C = np.cov(X.T, bias=False) if n > 1 else np.outer(X.ravel(), X.ravel())
            
            # Eigendecomposition to find optimal projection
            eigenvals, eigenvecs = np.linalg.eigh(C)
            
            # Sort in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Create rank-r projection operator H = Q_r Q_r^T
            Q_r = eigenvecs[:, :self.n_components]
            H = Q_r @ Q_r.T
            
            # Store components for later use
            self.components_ = Q_r.T  # Shape: (n_components, n_features)
            self.explained_variance_ = eigenvals[:self.n_components]
            self.explained_variance_ratio_ = (
                self.explained_variance_ / eigenvals.sum() 
                if eigenvals.sum() > 0 else np.zeros(self.n_components)
            )
            
            return H
        else:
            # For now, fall back to standard PCA
            # In future versions, this could implement other approximations
            return self._solve_via_standard_pca(X)
    
    def _solve_via_standard_pca(self, X: np.ndarray) -> np.ndarray:
        """Fallback to standard PCA approach."""
        # Use standard PCA as implemented in sklearn-style
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Select top r components
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (s[:self.n_components] ** 2) / (X.shape[0] - 1)
        total_var = (s ** 2).sum() / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0 
            else np.zeros(self.n_components)
        )
        
        # Create projection operator
        Q_r = self.components_.T  # Shape: (n_features, n_components)
        H = Q_r @ Q_r.T
        
        return H
    
    def _extract_components_from_operator(self, H: np.ndarray, X: np.ndarray) -> None:
        """Extract principal components from the fitted operator H."""
        # The components are already extracted in _solve_operator_choice_problem
        # This method is for consistency and future extensions
        pass
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted generalized PCA operator."""
        if not self.is_fitted:
            raise ValueError("GeneralizedPCA must be fitted before transforming")
        
        X = np.asarray(X)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        # Apply centering if used during fitting
        X_processed = X - self.mean_
        
        # Transform using components: X_transformed = X @ Q_r
        return X_processed @ self.components_.T
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from transformed space."""
        if not self.is_fitted:
            raise ValueError("GeneralizedPCA must be fitted before inverse transforming")
        
        # Reconstruct: X_reconstructed = X_transformed @ Q_r^T
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        return X_reconstructed + self.mean_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """For PCA, prediction is reconstruction."""
        X_transformed = self.transform(X)
        return self.inverse_transform(X_transformed)
    
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute performance metrics for generalized PCA."""
        if not self.is_fitted:
            raise ValueError("GeneralizedPCA must be fitted before scoring")
        
        X_reconstructed = self.predict(X)
        
        # Standard reconstruction metrics
        reconstruction_mse = np.mean((X - X_reconstructed) ** 2)
        explained_var_ratio = DataMetrics.explained_variance_ratio(X, X_reconstructed)
        
        metrics = {
            "reconstruction_mse": reconstruction_mse,
            "explained_variance_ratio": explained_var_ratio,
            "n_components": self.n_components,
            "total_explained_variance_ratio": self.explained_variance_ratio_.sum()
        }
        
        if y is not None:
            # If target is provided, compute prediction metrics
            # (though this is unusual for pure PCA)
            y = np.asarray(y)
            if y.ndim > 1:
                y = y.ravel()
            
            # For PCA, we can't predict y directly, so use mean as baseline
            y_pred = np.full(len(y), np.mean(y))
            metrics.update({
                "prediction_mse": DataMetrics.mse(y, y_pred),
                "prediction_r2": DataMetrics.r2_score(y, y_pred)
            })
        
        return metrics
    
    def get_operator_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the fitted operator."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "type": "GeneralizedPCA",
            "A_sigma": self.A_sigma,
            "A_beta": self.A_beta.copy(),
            "A_mu": self.A_mu.copy(),
            "n_components": self.n_components,
            "explained_variance": self.explained_variance_.copy(),
            "explained_variance_ratio": self.explained_variance_ratio_.copy(),
            "components": self.components_.copy(),
            "use_empirical_loss": self.use_empirical_loss,
            "center": self.center
        }
    

class GeneralizedPCARegressionOperator(GeneralizedPCAOperator):
    """
    True Generalized PCA with implicit weighting approach.
    
    This implements the unified operator from Chapter 4 that minimizes:
    ||X_tilde - A*X_tilde||_F^2 where X_tilde = [y, X]
    
    The operator A has the block structure:
    A = [A_σ  A_β^T]
        [A_β  A_μ  ]
    """
    
    def __init__(self, k: int, n_components: Optional[int] = None, **kwargs):
        """Initialize generalized PCA regression operator with implicit weighting."""
        super().__init__(k, n_components, **kwargs)
        self.intercept_ = 0.0
        # Storage for the unified operator matrix H
        self.H_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GeneralizedPCARegressionOperator':
        """
        Fit the generalized PCA using the implicit weighting approach.
        
        This implements the theoretical framework:
        minimize ||X_tilde - A*X_tilde||_F^2 where X_tilde = [y, X]
        and extracts the block structure A_σ, A_β, A_μ from the unified operator.
        """
        X, y = validate_data(X, y)
        
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.ravel()
        
        n, p = X.shape
        self.n_components = min(self.n_components, min(n, p + 1))
        
        # Preprocessing
        if self.center:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            y_centered = y - np.mean(y)
            self.intercept_ = np.mean(y)
        else:
            self.mean_ = np.zeros(p)
            X_centered = X.copy()
            y_centered = y.copy()
            self.intercept_ = 0.0
        
        # Apply the theoretical implicit weighting approach
        self._fit_implicit_weighting(X_centered, y_centered)
        
        self.is_fitted = True
        return self
    
    def _fit_implicit_weighting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the implicit weighting approach from the theoretical framework.
        
        Following the pseudocode:
        1. Create augmented matrix X_tilde = [y, X]
        2. Find H = argmin ||X_tilde - X_tilde @ H||_F^2 with rank(H) <= r
        3. Extract block structure A_σ, A_β, A_μ from H
        """
        n, p = X.shape
        
        # Step 1: Create augmented matrix X_tilde = [y, X]
        y_col = y.reshape(-1, 1)  # Ensure y is column vector
        X_tilde = np.hstack([y_col, X])  # Shape: (n, p+1)
        
        # Step 2: Find low-rank approximation via SVD
        # We want to minimize ||X_tilde - X_tilde @ H||_F^2
        # This is equivalent to finding the best rank-r approximation
        U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)
        
        # Create rank-r projection operator H
        V_r = Vt.T[:, :self.n_components]  # Shape: (p+1, r)
        self.H_ = V_r @ V_r.T               # Shape: (p+1, p+1)
        
        # Step 3: Extract block structure from H
        # H has structure:
        # H = [A_σ   A_β^T ]
        #     [A_β   A_μ   ]
        # where A_σ is scalar, A_β is (p,), A_μ is (p, p)
        
        self.A_sigma = self.H_[0, 0]         # Scalar
        self.A_beta = self.H_[1:, 0]         # Shape: (p,)
        self.A_mu = self.H_[1:, 1:]          # Shape: (p, p)
        
        # Store the principal components for compatibility
        # These are the components from the X part of the augmented space
        self.components_ = V_r[1:, :].T      # Shape: (r, p)
        self.explained_variance_ = S[:self.n_components] ** 2
        
        # Calculate explained variance ratios based on the augmented space
        total_var = np.sum(S ** 2)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0 
            else np.zeros(self.n_components)
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the unified operator structure.
        
        This applies the theoretical operator A as:
        y_pred = X @ A_β + A_σ * mean(y)
        
        where A_β and A_σ are extracted from the block structure of H.
        """
        if not self.is_fitted:
            raise ValueError("GeneralizedPCARegressionOperator must be fitted before predicting")
        
        X = np.asarray(X)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        # Apply centering if used during fitting
        X_processed = X - self.mean_
        
        # Apply the theoretical operator structure
        # y_pred = X @ A_β + A_σ * baseline
        y_pred = X_processed @ self.A_beta + self.intercept_
        
        return y_pred
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the component space defined by the unified operator.
        
        This projects onto the components extracted from the augmented space
        optimization but only for the X part.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        
        X = np.asarray(X)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        
        X_processed = X - self.mean_
        
        # Project onto the components from the augmented space optimization
        return X_processed @ self.components_.T
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute performance metrics for the implicit weighting generalized PCA."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.ravel()
        
        # Compute empirical MSE as approximation of model-theoretic loss L(μ, Hμ)
        prediction_mse = DataMetrics.mse(y, y_pred)
        prediction_r2 = DataMetrics.r2_score(y, y_pred)
        
        # Get base PCA metrics (reconstruction metrics for X)
        try:
            X_reconstructed = self.transform(X) @ self.components_ + self.mean_
            reconstruction_mse = np.mean((X - X_reconstructed) ** 2)
            explained_variance_ratio = 1 - reconstruction_mse / np.var(X)
        except Exception:
            reconstruction_mse = np.inf
            explained_variance_ratio = 0.0
        
        # Combine all metrics
        metrics = {
            "prediction_mse": prediction_mse,
            "prediction_r2": prediction_r2,
            "reconstruction_mse": reconstruction_mse,
            "explained_variance_ratio": explained_variance_ratio,
            "n_components": self.n_components,
            "total_explained_variance_ratio": (
                self.explained_variance_ratio_.sum() 
                if self.explained_variance_ratio_ is not None 
                else 0.0
            )
        }
        
        return metrics
    
    def get_operator_info(self) -> Dict[str, Any]:
        """Get information about the fitted implicit weighting operator."""
        info = super().get_operator_info()
        if self.is_fitted:
            info.update({
                "type": "GeneralizedPCARegression_ImplicitWeighting",
                "intercept": self.intercept_,
                "H_matrix": self.H_.copy() if self.H_ is not None else None,
                "A_sigma": float(self.A_sigma),
                "A_beta_norm": np.linalg.norm(self.A_beta),
                "A_mu_frobenius_norm": np.linalg.norm(self.A_mu, 'fro') if hasattr(self, 'A_mu') else 0.0,
                "theoretical_approach": "Implicit weighting via augmented matrix [y, X]"
            })
        return info