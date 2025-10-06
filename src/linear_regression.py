"""
Linear Regression implementation within the algebraic operator framework.

This module implements linear regression as a specific instance of operators in class P,
where A_μ acts like the identity and A_σ controls noise scaling.
"""

import numpy as np
from typing import Optional, Dict, Any
from .base import OperatorP, validate_data, DataMetrics


class LinearRegressionOperator(OperatorP):
    """
    Linear regression as an operator in class P.
    
    In the algebraic framework, linear regression corresponds to operators A ∈ P
    where:
    - A_μ is (or acts like) the identity on the predictor block
    - A_β gives the regression coefficients  
    - A_σ controls the noise scale
    
    The response follows: y = Σ A_β,i * X_i + A_σ * ε
    """
    
    def __init__(self, k: int, fit_intercept: bool = True, regularization: Optional[str] = None, 
                 alpha: float = 1.0):
        """
        Initialize linear regression operator.
        
        Parameters:
        k: Dimension of augmented space
        fit_intercept: Whether to fit an intercept term
        regularization: Type of regularization ('ridge', 'lasso', None)
        alpha: Regularization strength
        """
        super().__init__(k)
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.alpha = alpha
        self.intercept_ = 0.0
        self.is_fitted = False
        
        # A_μ remains identity for standard linear regression
        self.A_mu = np.eye(self.p)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionOperator':
        """
        Fit linear regression operator to data.
        
        Parameters:
        X: Feature matrix of shape (n, p)
        y: Target vector of shape (n,)
        
        Returns:
        Self (fitted operator)
        """
        X, y = validate_data(X, y)
        n, p = X.shape
        
        if p != self.p:
            raise ValueError(f"Expected {self.p} features, got {p}")
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(n), X])
        else:
            X_with_intercept = X
        
        # Fit regression coefficients
        if self.regularization is None:
            # Ordinary least squares
            coefficients = self._fit_ols(X_with_intercept, y)
        elif self.regularization == 'ridge':
            coefficients = self._fit_ridge(X_with_intercept, y)
        else:
            raise NotImplementedError(f"Regularization {self.regularization} not implemented")
        
        # Extract intercept and slopes
        if self.fit_intercept:
            self.intercept_ = coefficients[0]
            self.A_beta = coefficients[1:]
        else:
            self.intercept_ = 0.0
            self.A_beta = coefficients
        
        self.is_fitted = True  # Set fitted flag before prediction
        
        # Estimate noise scaling (A_σ) from residuals
        y_pred = self.predict(X)
        residuals = y - y_pred
        self.A_sigma = np.std(residuals)
        
        return self
    
    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit using ordinary least squares."""
        # β = (X^T X)^(-1) X^T y
        try:
            return np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if X^T X is singular
            return np.linalg.pinv(X) @ y
    
    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit using ridge regression."""
        # β = (X^T X + αI)^(-1) X^T y
        n_features = X.shape[1]
        I = np.eye(n_features)
        if self.fit_intercept:
            # Don't regularize intercept
            I[0, 0] = 0
        return np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted operator.
        
        Parameters:
        X: Feature matrix of shape (n, p)
        
        Returns:
        Predictions of shape (n,)
        """
        if not self.is_fitted:
            raise ValueError("Operator must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.p:
            raise ValueError(f"X must have shape (n, {self.p})")
        
        # y = A_β^T X + intercept (noise term A_σ * ε excluded for prediction)
        return X @ self.A_beta + self.intercept_
    
    def predict_with_noise(self, X: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        """
        Make predictions including noise term A_σ * ε.
        
        This corresponds to the full operator response y = (AZ)_1.
        """
        if not self.is_fitted:
            raise ValueError("Operator must be fitted before making predictions")
        
        base_prediction = self.predict(X)
        return base_prediction + self.A_sigma * epsilon
    
    def get_operator_info(self) -> Dict[str, Any]:
        """Get information about the fitted operator."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "type": "LinearRegression",
            "A_sigma": self.A_sigma,
            "A_beta": self.A_beta.copy(),
            "A_mu": self.A_mu.copy(),
            "intercept": self.intercept_,
            "n_features": self.p,
            "regularization": self.regularization,
            "alpha": self.alpha if self.regularization else None
        }
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute various performance metrics.
        
        Returns:
        Dictionary with MSE, R², and other metrics
        """
        y_pred = self.predict(X)
        
        return {
            "mse": DataMetrics.mse(y, y_pred),
            "r2": DataMetrics.r2_score(y, y_pred),
            "rmse": np.sqrt(DataMetrics.mse(y, y_pred)),
            "mae": np.mean(np.abs(y - y_pred))
        }


class LinearRegressionFamily:
    """
    Collection of linear regression operators corresponding to subsemirings L_ℓ and S_ℓ.
    
    This implements the algebraic variable selection framework described in the theory,
    where:
    - S_ℓ: operators using only the ℓ-th predictor
    - L_ℓ: operators using predictors 1, ..., ℓ
    """
    
    def __init__(self, k: int, **kwargs):
        """
        Initialize family of linear regression operators.
        
        Parameters:
        k: Dimension of augmented space
        **kwargs: Additional parameters passed to LinearRegressionOperator
        """
        self.k = k
        self.p = k - 1
        self.kwargs = kwargs
        self.operators = {}
    
    def fit_S_ell(self, X: np.ndarray, y: np.ndarray, ell: int) -> LinearRegressionOperator:
        """
        Fit operator from subsemiring S_ℓ (using only ℓ-th predictor).
        
        Parameters:
        X: Full feature matrix
        y: Target vector
        ell: Index of predictor to use (1-indexed)
        
        Returns:
        Fitted operator using only the ell-th predictor
        """
        if not (1 <= ell <= self.p):
            raise ValueError(f"ell must be between 1 and {self.p}")
        
        # Use only the ell-th predictor (convert to 0-indexed)
        X_ell = X[:, [ell-1]]
        
        # Create operator with appropriate dimension
        operator = LinearRegressionOperator(2, **self.kwargs)  # k=2 for single predictor
        operator.fit(X_ell, y)
        
        # Store in family
        self.operators[f"S_{ell}"] = operator
        return operator
    
    def fit_L_ell(self, X: np.ndarray, y: np.ndarray, ell: int) -> LinearRegressionOperator:
        """
        Fit operator from subsemiring L_ℓ (using predictors 1, ..., ℓ).
        
        Parameters:
        X: Full feature matrix
        y: Target vector
        ell: Number of predictors to use
        
        Returns:
        Fitted operator using first ell predictors
        """
        if not (1 <= ell <= self.p):
            raise ValueError(f"ell must be between 1 and {self.p}")
        
        # Use first ell predictors
        X_ell = X[:, :ell]
        
        # Create operator with appropriate dimension
        operator = LinearRegressionOperator(ell + 1, **self.kwargs)
        operator.fit(X_ell, y)
        
        # Store in family
        self.operators[f"L_{ell}"] = operator
        return operator
    
    def fit_all_S(self, X: np.ndarray, y: np.ndarray) -> Dict[str, LinearRegressionOperator]:
        """Fit all S_ℓ operators."""
        for ell in range(1, self.p + 1):
            self.fit_S_ell(X, y, ell)
        return {k: v for k, v in self.operators.items() if k.startswith("S_")}
    
    def fit_all_L(self, X: np.ndarray, y: np.ndarray) -> Dict[str, LinearRegressionOperator]:
        """Fit all L_ℓ operators."""
        for ell in range(1, self.p + 1):
            self.fit_L_ell(X, y, ell)
        return {k: v for k, v in self.operators.items() if k.startswith("L_")}
    
    def compare_operators(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of all fitted operators.
        
        Returns:
        Dictionary mapping operator names to their performance metrics
        """
        results = {}
        
        for name, operator in self.operators.items():
            # Adjust test data to match operator's feature dimension
            if name.startswith("S_"):
                ell = int(name.split("_")[1])
                X_test_adj = X_test[:, [ell-1]]
            elif name.startswith("L_"):
                ell = int(name.split("_")[1])
                X_test_adj = X_test[:, :ell]
            else:
                X_test_adj = X_test
            
            results[name] = operator.score(X_test_adj, y_test)
        
        return results