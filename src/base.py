"""
Base classes and common utilities for the algebraic framework implementation.
This module provides the foundational structures for representing operators
in the class P as described in the theoretical framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import warnings


class OperatorP(ABC):
    """
    Base class for operators in the class P as defined in equation (4.1).
    
    An operator A ∈ P has the block structure:
    A = [A_σ    A_β   ]
        [0      A_μ   ]
    
    where:
    - A_σ ≥ 0: scalar noise scaling factor
    - A_β ∈ ℝ^(1×(k-1)): regression coefficients
    - A_μ ∈ ℝ^((k-1)×(k-1)): predictor transformation matrix
    """
    
    def __init__(self, k: int):
        """
        Initialize operator for k-dimensional augmented space.
        
        Parameters:
        k: Dimension of augmented vector Z = (ε, X_1, ..., X_{k-1})
        """
        self.k = k
        self.p = k - 1  # Number of predictors
        self.A_sigma = 0.0
        self.A_beta = np.zeros(self.p)
        self.A_mu = np.eye(self.p)
        
    @property
    def matrix(self) -> np.ndarray:
        """Return the full block matrix representation of the operator."""
        A = np.zeros((self.k, self.k))
        A[0, 0] = self.A_sigma
        A[0, 1:] = self.A_beta
        A[1:, 1:] = self.A_mu
        return A
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OperatorP':
        """Fit the operator to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted operator."""
        pass
    
    def response(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the response y = (AZ)_1 as in equation (4.2).
        
        Parameters:
        Z: Augmented vector (ε, X_1, ..., X_{k-1}) of shape (n, k)
        
        Returns:
        Response vector of shape (n,)
        """
        if Z.shape[1] != self.k:
            raise ValueError(f"Z must have {self.k} columns, got {Z.shape[1]}")
        
        # y = Σ A_β,i * X_i + A_σ * ε
        # Assuming ε is the first column and X_i are the remaining columns
        epsilon = Z[:, 0]
        X = Z[:, 1:]
        
        y = np.dot(X, self.A_beta) + self.A_sigma * epsilon
        return y
    
    def __add__(self, other: 'OperatorP') -> 'OperatorP':
        """Implement operator addition (⊕ operation in the semiring)."""
        if not isinstance(other, OperatorP):
            raise TypeError("Can only add OperatorP instances")
        if self.k != other.k:
            raise ValueError("Operators must have same dimension")
            
        result = type(self)(self.k)
        result.A_sigma = self.A_sigma + other.A_sigma
        result.A_beta = self.A_beta + other.A_beta
        result.A_mu = self.A_mu + other.A_mu
        return result
    
    def __matmul__(self, other: 'OperatorP') -> 'OperatorP':
        """Implement operator multiplication (⊗ operation in the semiring)."""
        if not isinstance(other, OperatorP):
            raise TypeError("Can only multiply OperatorP instances")
        if self.k != other.k:
            raise ValueError("Operators must have same dimension")
            
        # Block matrix multiplication
        # [A_σ A_β] [B_σ B_β] = [A_σB_σ  A_σB_β + A_βB_μ]
        # [0   A_μ] [0   B_μ]   [0      A_μB_μ          ]
        
        result = type(self)(self.k)
        result.A_sigma = self.A_sigma * other.A_sigma
        result.A_beta = self.A_sigma * other.A_beta + np.dot(self.A_beta, other.A_mu)
        result.A_mu = np.dot(self.A_mu, other.A_mu)
        return result


class DataMetrics:
    """Utility class for computing various error metrics and consistency measures."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    @staticmethod
    def operator_stability(H1: np.ndarray, H2: np.ndarray, norm: str = 'fro') -> float:
        """
        Measure operator stability ||H^(n) - H^(n+m)||_F.
        
        This quantifies how much the operator changes when sample size increases,
        which is crucial for measuring extendability as described in the theory.
        """
        if norm == 'fro':
            return np.linalg.norm(H1 - H2, 'fro')
        elif norm == 'spectral':
            return np.linalg.norm(H1 - H2, 2)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    @staticmethod
    def explained_variance_ratio(X: np.ndarray, X_reconstructed: np.ndarray) -> float:
        """Compute explained variance ratio for reconstruction methods."""
        total_var = np.var(X, axis=0).sum()
        residual_var = np.var(X - X_reconstructed, axis=0).sum()
        return 1 - (residual_var / total_var) if total_var != 0 else 0.0


def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and prepare data for operator fitting.
    
    Parameters:
    X: Feature matrix of shape (n, p)
    y: Target vector of shape (n,)
    
    Returns:
    Validated and properly shaped X and y
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    
    # Check for NaN or infinite values
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values")
    
    return X, y


def create_augmented_vector(X: np.ndarray, epsilon: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create augmented vector Z = (ε, X_1, ..., X_{k-1}) for operator application.
    
    Parameters:
    X: Feature matrix of shape (n, p)
    epsilon: Noise vector of shape (n,). If None, zeros are used.
    
    Returns:
    Augmented matrix Z of shape (n, k) where k = p + 1
    """
    n, p = X.shape
    
    if epsilon is None:
        epsilon = np.zeros(n)
    else:
        epsilon = np.asarray(epsilon)
        if epsilon.shape != (n,):
            raise ValueError(f"epsilon must have shape ({n},), got {epsilon.shape}")
    
    Z = np.column_stack([epsilon, X])
    return Z