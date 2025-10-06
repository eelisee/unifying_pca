"""
Unifying PCA: Algebraic Framework Implementation

This package implements the algebraic framework for unifying linear regression
and PCA as described in the theoretical work. It provides:

1. Base classes for operators in the semiring P
2. Linear regression as operators with identity A_μ
3. Standard PCA as projection operators  
4. Generalized PCA using model-theoretic optimization
5. Comprehensive comparison framework

The implementation follows the theoretical framework where both regression
and PCA are viewed as operators acting on augmented data vectors, with
different constraints on the operator structure.
"""

from .base import OperatorP, DataMetrics, validate_data, create_augmented_vector
from .linear_regression import LinearRegressionOperator, LinearRegressionFamily
from .pca import PCAOperator, PCARegressionOperator, PCAFamily
from .generalized_pca import GeneralizedPCAOperator, GeneralizedPCARegressionOperator
from .comparison import ModelComparison, run_dataset_comparison

__version__ = "1.0.0"
__author__ = "Unifying PCA Research Team"

__all__ = [
    # Base classes
    'OperatorP',
    'DataMetrics',
    'validate_data',
    'create_augmented_vector',
    
    # Linear Regression
    'LinearRegressionOperator',
    'LinearRegressionFamily',
    
    # Standard PCA
    'PCAOperator',
    'PCARegressionOperator', 
    'PCAFamily',
    
    # Generalized PCA
    'GeneralizedPCAOperator',
    'GeneralizedPCARegressionOperator',
    
    # Comparison Framework
    'ModelComparison',
    'run_dataset_comparison'
]

def get_version():
    """Return the current version."""
    return __version__

def get_available_operators():
    """Return a list of available operator types."""
    return [
        'LinearRegressionOperator',
        'PCAOperator', 
        'PCARegressionOperator',
        'GeneralizedPCAOperator',
        'GeneralizedPCARegressionOperator'
    ]

def create_operator(operator_type: str, k: int, **kwargs):
    """
    Factory function to create operators of different types.
    
    Parameters:
    operator_type: Type of operator to create
    k: Dimension of augmented space
    **kwargs: Additional parameters for the operator
    
    Returns:
    Instantiated operator
    """
    operator_map = {
        'linear_regression': LinearRegressionOperator,
        'pca': PCAOperator,
        'pca_regression': PCARegressionOperator,
        'generalized_pca': GeneralizedPCAOperator,
        'generalized_pca_regression': GeneralizedPCARegressionOperator
    }
    
    if operator_type not in operator_map:
        raise ValueError(f"Unknown operator type: {operator_type}. "
                        f"Available types: {list(operator_map.keys())}")
    
    return operator_map[operator_type](k, **kwargs)