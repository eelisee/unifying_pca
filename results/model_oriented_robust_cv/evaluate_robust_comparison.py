"""
Robust Evaluation of Model-Oriented Generalized PCA
Focus on numerical stability and comparative analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
from datetime import datetime
import sys

# Add src to path
sys.path.append('src')

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from src.model_oriented_generalized_pca import ModelOrientedGeneralizedPCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RobustModelEvaluator:
    """
    Robust evaluator with improved numerical stability and focused analysis.
    Includes k-fold cross-validation and KL divergence for all models.
    """
    
    def __init__(self, data_dir: str = "data/processed", results_dir: str = "results/model_oriented_robust", 
                 k_folds: int = 5):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.k_folds = k_folds
        
        # Results storage
        self.all_results = []
        self.failed_experiments = []
        
        # Select stable datasets for focused evaluation
        self.stable_datasets = [
            'diabetes', 'california_housing', 'concrete_strength', 
            'wine_quality_red', 'wine_quality_white', 'energy_efficiency'
        ]
    
    def _load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load a dataset and its metadata."""
        dataset_path = self.data_dir / dataset_name
        
        # Load data
        X_train = pd.read_csv(dataset_path / 'X_train_scaled.csv').values
        y_train = pd.read_csv(dataset_path / 'y_train.csv').values.ravel()
        X_test = pd.read_csv(dataset_path / 'X_test_scaled.csv').values
        y_test = pd.read_csv(dataset_path / 'y_test.csv').values.ravel()
        
        # Load metadata if available
        metadata_path = dataset_path / 'metadata.json'
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"Loaded {dataset_name}: X_train {X_train.shape}, y_train {y_train.shape}")
        
        return X_train, y_train, X_test, y_test, metadata
    
    def _compute_kl_divergence(self, Sigma_emp: np.ndarray, Sigma_model: np.ndarray, regularization: float = 1e-8) -> float:
        """Compute KL divergence between two multivariate Gaussians."""
        m = Sigma_emp.shape[0]
        
        # Add regularization for numerical stability
        Sigma_model_reg = Sigma_model + regularization * np.eye(m)
        
        try:
            # Compute inverse and determinants
            Sigma_model_inv = np.linalg.inv(Sigma_model_reg)
            
            # Use slogdet for numerical stability
            sign_emp, logdet_emp = np.linalg.slogdet(Sigma_emp)
            sign_model, logdet_model = np.linalg.slogdet(Sigma_model_reg)
            
            if sign_emp <= 0 or sign_model <= 0:
                return np.inf
            
            # KL divergence formula
            trace_term = np.trace(Sigma_model_inv @ Sigma_emp)
            det_term = logdet_model - logdet_emp
            kl = 0.5 * (trace_term - m + det_term)
            
            return kl
            
        except (np.linalg.LinAlgError):
            return np.inf
    
    def _build_lr_covariance_matrix(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray, sigma: float) -> np.ndarray:
        """
        Build model-induced covariance matrix for Linear Regression.
        
        For Z = [X; y], the LR model implies:
        Σ_Z^LR = [[Σ_X, Σ_X β], [β^T Σ_X, β^T Σ_X β + σ²]]
        """
        n, p = X.shape
        
        # Empirical covariance of X
        Sigma_X = (X.T @ X) / n
        
        # Build model-induced covariance
        Sigma_Z_lr = np.zeros((p + 1, p + 1))
        
        # Top-left: Σ_X
        Sigma_Z_lr[:p, :p] = Sigma_X
        
        # Top-right and bottom-left: Σ_X β
        Sigma_X_beta = Sigma_X @ beta.reshape(-1, 1)
        Sigma_Z_lr[:p, p] = Sigma_X_beta.flatten()
        Sigma_Z_lr[p, :p] = Sigma_X_beta.flatten()
        
        # Bottom-right: β^T Σ_X β + σ²
        Sigma_Z_lr[p, p] = float(beta.T @ Sigma_X @ beta + sigma**2)
        
        return Sigma_Z_lr
    
    def _build_pca_covariance_matrix(self, X: np.ndarray, pca_components: np.ndarray, 
                                   explained_variance: np.ndarray, noise_variance: float) -> np.ndarray:
        """
        Build model-induced covariance matrix for PCA.
        
        Σ_X^PCA = V_r Λ_r V_r^T + σ² I
        """
        p = X.shape[1]
        r = len(explained_variance)
        
        # Reconstruct covariance from PCA
        V_r = pca_components[:r, :].T  # p x r
        Lambda_r = np.diag(explained_variance)  # r x r
        
        # Model-induced covariance
        Sigma_X_pca = V_r @ Lambda_r @ V_r.T + noise_variance * np.eye(p)
        
        return Sigma_X_pca
    
    def _cross_validate_model(self, X: np.ndarray, y: np.ndarray, model_type: str, **model_kwargs) -> Dict[str, Any]:
        """Perform k-fold cross-validation for a model."""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'cv_scores': [],
            'cv_mse': [],
            'cv_kl_divergence': [],
            'fold_results': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            try:
                if model_type == 'linear_regression':
                    fold_result = self._evaluate_lr_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                elif model_type == 'pca':
                    n_components = model_kwargs.get('n_components', min(X.shape[1], X.shape[0] - 1, 6))
                    fold_result = self._evaluate_pca_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, n_components)
                elif model_type == 'mogpca':
                    fold_result = self._evaluate_mogpca_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **model_kwargs)
                else:
                    continue
                
                cv_results['cv_scores'].append(fold_result.get('r2', np.nan))
                cv_results['cv_mse'].append(fold_result.get('mse', np.nan))
                cv_results['cv_kl_divergence'].append(fold_result.get('kl_divergence', np.nan))
                cv_results['fold_results'].append(fold_result)
                
            except Exception as e:
                print(f"    Fold {fold} failed: {e}")
                cv_results['cv_scores'].append(np.nan)
                cv_results['cv_mse'].append(np.nan)
                cv_results['cv_kl_divergence'].append(np.nan)
        
        # Compute CV statistics
        valid_scores = [s for s in cv_results['cv_scores'] if not np.isnan(s)]
        valid_mse = [s for s in cv_results['cv_mse'] if not np.isnan(s)]
        valid_kl = [s for s in cv_results['cv_kl_divergence'] if not np.isnan(s) and np.isfinite(s)]
        
        cv_results.update({
            'cv_score_mean': np.mean(valid_scores) if valid_scores else np.nan,
            'cv_score_std': np.std(valid_scores) if valid_scores else np.nan,
            'cv_mse_mean': np.mean(valid_mse) if valid_mse else np.nan,
            'cv_mse_std': np.std(valid_mse) if valid_mse else np.nan,
            'cv_kl_mean': np.mean(valid_kl) if valid_kl else np.nan,
            'cv_kl_std': np.std(valid_kl) if valid_kl else np.nan,
            'n_successful_folds': len(valid_scores)
        })
        
        return cv_results
    
    def _evaluate_lr_fold(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Evaluate Linear Regression on a single fold."""
        # Fit model
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = lr.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Compute KL divergence
        residuals = y_train - lr.predict(X_train)
        sigma = np.std(residuals)
        
        # Build empirical covariance for [X; y]
        Z_train = np.column_stack([X_train, y_train.reshape(-1, 1)])
        Sigma_Z_emp = np.cov(Z_train.T, bias=True)
        
        # Build LR model covariance
        Sigma_Z_lr = self._build_lr_covariance_matrix(X_train, y_train, lr.coef_, sigma)
        
        # Compute KL divergence
        kl_divergence = self._compute_kl_divergence(Sigma_Z_emp, Sigma_Z_lr)
        
        return {
            'mse': mse,
            'r2': r2,
            'kl_divergence': kl_divergence,
            'sigma': sigma,
            'beta_norm': np.linalg.norm(lr.coef_)
        }
    
    def _evaluate_pca_fold(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray, n_components: int) -> Dict[str, Any]:
        """Evaluate PCA on a single fold."""
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        
        # Transform and predict
        X_train_transformed = pca.transform(X_train)
        X_val_transformed = pca.transform(X_val)
        
        # Fit regression on PCA components
        lr_pca = LinearRegression(fit_intercept=False)
        lr_pca.fit(X_train_transformed, y_train)
        
        y_pred = lr_pca.predict(X_val_transformed)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Compute KL divergence for PCA model
        # Build empirical covariance of X
        Sigma_X_emp = np.cov(X_train.T, bias=True)
        
        # Estimate noise variance from reconstruction error
        X_train_reconstructed = pca.inverse_transform(X_train_transformed)
        reconstruction_errors = X_train - X_train_reconstructed
        noise_variance = np.mean(np.var(reconstruction_errors, axis=0))
        
        # Build PCA model covariance
        Sigma_X_pca = self._build_pca_covariance_matrix(X_train, pca.components_, 
                                                       pca.explained_variance_, noise_variance)
        
        # Compute KL divergence
        kl_divergence = self._compute_kl_divergence(Sigma_X_emp, Sigma_X_pca)
        
        return {
            'mse': mse,
            'r2': r2,
            'kl_divergence': kl_divergence,
            'noise_variance': noise_variance,
            'explained_variance_ratio': np.sum(pca.explained_variance_ratio_)
        }
    
    def _evaluate_mogpca_fold(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray, **model_kwargs) -> Dict[str, Any]:
        """Evaluate Model-Oriented Generalized PCA on a single fold."""
        # Use conservative parameters for CV
        max_r = min(X_train.shape[1], X_train.shape[0] - 1, 3)
        r_beta_candidates = list(range(0, min(3, max_r + 1)))
        r_H_candidates = list(range(1, max_r + 1))
        
        mogpca = ModelOrientedGeneralizedPCA(
            r_beta_candidates=r_beta_candidates,
            r_H_candidates=r_H_candidates,
            max_iter=10,  # Fewer iterations for CV
            tol=1e-4,
            model_selection='bic',
            regularization=1e-5,
            random_state=42
        )
        
        # Fit and predict
        mogpca.fit(X_train, y_train)
        y_pred = mogpca.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        kl_divergence = mogpca.kl_divergence_
        
        return {
            'mse': mse,
            'r2': r2,
            'kl_divergence': kl_divergence if np.isfinite(kl_divergence) else np.nan,
            'r_beta_optimal': mogpca.r_beta_optimal_,
            'r_H_optimal': mogpca.r_H_optimal_
        }
    
    def _evaluate_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate Linear Regression model with cross-validation and KL divergence."""
        print("  Evaluating Linear Regression...")
        
        # Cross-validation
        cv_results = self._cross_validate_model(X_train, y_train, 'linear_regression')
        
        # Fit OLS on full training data
        ols = LinearRegression(fit_intercept=False)  # Data already centered
        ols.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = ols.predict(X_train)
        y_pred_test = ols.predict(X_test)
        
        # Data-level metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Baseline MSE (predicting mean)
        baseline_train_mse = mean_squared_error(y_train, np.full_like(y_train, np.mean(y_train)))
        baseline_test_mse = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)))
        
        # Relative MSE improvement over baseline
        train_mse_improvement = (baseline_train_mse - train_mse) / baseline_train_mse
        test_mse_improvement = (baseline_test_mse - test_mse) / baseline_test_mse
        
        # Residuals analysis
        residuals = y_train - y_pred_train
        residual_std = np.std(residuals)
        
        # Compute KL divergence on full training data
        sigma = np.std(residuals)
        Z_train = np.column_stack([X_train, y_train.reshape(-1, 1)])
        Sigma_Z_emp = np.cov(Z_train.T, bias=True)
        Sigma_Z_lr = self._build_lr_covariance_matrix(X_train, y_train, ols.coef_, sigma)
        kl_divergence = self._compute_kl_divergence(Sigma_Z_emp, Sigma_Z_lr)
        
        result = {
            'model': 'LinearRegression',
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'baseline_train_mse': baseline_train_mse,
            'baseline_test_mse': baseline_test_mse,
            'train_mse_improvement': train_mse_improvement,
            'test_mse_improvement': test_mse_improvement,
            'residual_std': residual_std,
            'coefficients_norm': np.linalg.norm(ols.coef_),
            'kl_divergence': kl_divergence
        }
        
        # Add CV results
        result.update(cv_results)
        
        return result
    
    def _evaluate_pca(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray, n_components: int = None) -> Dict[str, Any]:
        """Evaluate PCA model with cross-validation and KL divergence."""
        print("  Evaluating PCA...")
        
        # Determine number of components
        if n_components is None:
            n_components = min(X_train.shape[1], X_train.shape[0] - 1, 6)
        
        # Cross-validation
        cv_results = self._cross_validate_model(X_train, y_train, 'pca', n_components=n_components)
        
        # Fit PCA on full training data
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        
        # Transform and reconstruct
        X_train_transformed = pca.transform(X_train)
        X_test_transformed = pca.transform(X_test)
        X_train_reconstructed = pca.inverse_transform(X_train_transformed)
        X_test_reconstructed = pca.inverse_transform(X_test_transformed)
        
        # Data-level metrics (reconstruction error)
        train_reconstruction_mse = np.mean((X_train - X_train_reconstructed)**2)
        test_reconstruction_mse = np.mean((X_test - X_test_reconstructed)**2)
        
        # For Y prediction, fit regression on PCA components
        ols_on_pca = LinearRegression(fit_intercept=False)
        ols_on_pca.fit(X_train_transformed, y_train)
        
        y_pred_train = ols_on_pca.predict(X_train_transformed)
        y_pred_test = ols_on_pca.predict(X_test_transformed)
        
        train_pred_mse = mean_squared_error(y_train, y_pred_train)
        test_pred_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Baseline MSE (predicting mean)
        baseline_train_mse = mean_squared_error(y_train, np.full_like(y_train, np.mean(y_train)))
        baseline_test_mse = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)))
        
        # Relative MSE improvement over baseline
        train_mse_improvement = (baseline_train_mse - train_pred_mse) / baseline_train_mse
        test_mse_improvement = (baseline_test_mse - test_pred_mse) / baseline_test_mse
        
        # Compute KL divergence
        Sigma_X_emp = np.cov(X_train.T, bias=True)
        reconstruction_errors = X_train - X_train_reconstructed
        noise_variance = np.mean(np.var(reconstruction_errors, axis=0))
        Sigma_X_pca = self._build_pca_covariance_matrix(X_train, pca.components_, 
                                                       pca.explained_variance_, noise_variance)
        kl_divergence = self._compute_kl_divergence(Sigma_X_emp, Sigma_X_pca)
        
        result = {
            'model': 'PCA',
            'n_components': n_components,
            'train_reconstruction_mse': train_reconstruction_mse,
            'test_reconstruction_mse': test_reconstruction_mse,
            'train_mse': train_pred_mse,
            'test_mse': test_pred_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'baseline_train_mse': baseline_train_mse,
            'baseline_test_mse': baseline_test_mse,
            'train_mse_improvement': train_mse_improvement,
            'test_mse_improvement': test_mse_improvement,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_explained': np.sum(pca.explained_variance_ratio_),
            'kl_divergence': kl_divergence,
            'noise_variance': noise_variance
        }
        
        # Add CV results
        result.update(cv_results)
        
        return result
    
    def _evaluate_generalized_pca(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate Model-Oriented Generalized PCA with robust parameters and cross-validation."""
        print("  Evaluating Model-Oriented Generalized PCA...")
        
        # Use conservative parameters for numerical stability
        max_r = min(X_train.shape[1], X_train.shape[0] - 1, 4)  # Smaller ranges
        r_beta_candidates = list(range(0, min(3, max_r + 1)))  # Limit to 0,1,2
        r_H_candidates = list(range(1, max_r + 1))  # Limit components
        
        # Cross-validation
        cv_results = self._cross_validate_model(X_train, y_train, 'mogpca',
                                              r_beta_candidates=r_beta_candidates,
                                              r_H_candidates=r_H_candidates)
        
        mogpca = ModelOrientedGeneralizedPCA(
            r_beta_candidates=r_beta_candidates,
            r_H_candidates=r_H_candidates,
            max_iter=15,  # Fewer iterations
            tol=1e-5,     # Less strict tolerance
            model_selection='bic',
            regularization=1e-5,  # Higher regularization for stability
            random_state=42
        )
        
        try:
            # Fit model
            mogpca.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = mogpca.predict(X_train)
            y_pred_test = mogpca.predict(X_test)
            
            # Data-level metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Baseline MSE (predicting mean)
            baseline_train_mse = mean_squared_error(y_train, np.full_like(y_train, np.mean(y_train)))
            baseline_test_mse = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)))
            
            # Relative MSE improvement over baseline
            train_mse_improvement = (baseline_train_mse - train_mse) / baseline_train_mse
            test_mse_improvement = (baseline_test_mse - test_mse) / baseline_test_mse
            
            # Model-level metrics
            kl_divergence = mogpca.kl_divergence_
            is_valid_kl = np.isfinite(kl_divergence) and kl_divergence >= 0
            
            # Get diagnostics
            diagnostics = mogpca.get_diagnostics()
            
            result = {
                'model': 'ModelOrientedGeneralizedPCA',
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'baseline_train_mse': baseline_train_mse,
                'baseline_test_mse': baseline_test_mse,
                'train_mse_improvement': train_mse_improvement,
                'test_mse_improvement': test_mse_improvement,
                'kl_divergence': kl_divergence if is_valid_kl else np.nan,
                'r_beta_optimal': mogpca.r_beta_optimal_,
                'r_H_optimal': mogpca.r_H_optimal_,
                'n_iterations': mogpca.n_iter_,
                'aic_score': mogpca.aic_score_,
                'bic_score': mogpca.bic_score_,
                'residual_scale': mogpca.residual_scale_,
                'is_valid_kl': is_valid_kl,
                'status': 'success'
            }
            
            # Add CV results
            result.update(cv_results)
            
            return result
            
        except Exception as e:
            print(f"    Error in MOGPCA: {e}")
            result = {
                'model': 'ModelOrientedGeneralizedPCA',
                'train_mse': np.nan,
                'test_mse': np.nan,
                'train_r2': np.nan,
                'test_r2': np.nan,
                'kl_divergence': np.nan,
                'error': str(e),
                'status': 'failed'
            }
            
            # Add empty CV results for consistency
            result.update({
                'cv_score_mean': np.nan,
                'cv_score_std': np.nan,
                'cv_mse_mean': np.nan,
                'cv_mse_std': np.nan,
                'cv_kl_mean': np.nan,
                'cv_kl_std': np.nan,
                'n_successful_folds': 0
            })
            
            return result
    
    def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Evaluate all models on a single dataset."""
        print(f"\nEvaluating dataset: {dataset_name}")
        print("=" * 50)
        
        # Load dataset
        X_train, y_train, X_test, y_test, metadata = self._load_dataset(dataset_name)
        
        # Initialize results
        dataset_results = {
            'dataset': dataset_name,
            'metadata': metadata,
            'data_info': {
                'n_train_samples': X_train.shape[0],
                'n_test_samples': X_test.shape[0],
                'n_features': X_train.shape[1],
                'train_target_mean': float(np.mean(y_train)),
                'train_target_std': float(np.std(y_train)),
                'test_target_mean': float(np.mean(y_test)),
                'test_target_std': float(np.std(y_test)),
                'feature_matrix_condition': float(np.linalg.cond(X_train))
            },
            'models': {}
        }
        
        # Evaluate each model
        try:
            # 1. Linear Regression
            lr_results = self._evaluate_linear_regression(X_train, y_train, X_test, y_test)
            dataset_results['models']['LinearRegression'] = lr_results
            
            # 2. PCA
            pca_results = self._evaluate_pca(X_train, y_train, X_test, y_test)
            dataset_results['models']['PCA'] = pca_results
            
            # 3. Model-Oriented Generalized PCA
            mogpca_results = self._evaluate_generalized_pca(X_train, y_train, X_test, y_test)
            dataset_results['models']['ModelOrientedGeneralizedPCA'] = mogpca_results
            
            # Summary comparison
            print("\\nResults Summary:")
            print("-" * 30)
            for model_name, results in dataset_results['models'].items():
                if 'test_mse' in results and not np.isnan(results['test_mse']):
                    test_mse = results['test_mse']
                    baseline_mse = results.get('baseline_test_mse', np.nan)
                    mse_improvement = results.get('test_mse_improvement', np.nan)
                    kl_str = f"{results.get('kl_divergence', np.nan):10.6f}" if 'kl_divergence' in results else "     N/A"
                    
                    if results.get('is_valid_kl', True):  # Only show if valid
                        print(f"{model_name:25s}: MSE = {test_mse:8.4f} (vs baseline {baseline_mse:8.4f}), KL = {kl_str}")
                    else:
                        print(f"{model_name:25s}: MSE = {test_mse:8.4f} (vs baseline {baseline_mse:8.4f}), KL = invalid")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            self.failed_experiments.append({'dataset': dataset_name, 'error': str(e)})
            return None
        
        return dataset_results
    
    def run_robust_evaluation(self):
        """Run evaluation on stable datasets with robust parameters."""
        print("Starting Robust Model Comparison Evaluation")
        print("=" * 60)
        print("Comparing: Linear Regression, PCA, Model-Oriented Generalized PCA")
        print("Focus: Numerical Stability and Interpretable Results")
        print(f"Target datasets: {', '.join(self.stable_datasets)}")
        print()
        
        for dataset_name in self.stable_datasets:
            # Check if dataset exists
            if not (self.data_dir / dataset_name).exists():
                print(f"Dataset {dataset_name} not found, skipping...")
                continue
                
            try:
                dataset_result = self.evaluate_dataset(dataset_name)
                if dataset_result is not None:
                    self.all_results.append(dataset_result)
            except Exception as e:
                print(f"Failed to evaluate {dataset_name}: {e}")
                self.failed_experiments.append({'dataset': dataset_name, 'error': str(e)})
                continue
        
        # Save detailed results
        self._save_results()
        
        # Create summary and visualizations
        self._create_summary()
        self._create_visualizations()
        
        print(f"\\nRobust Evaluation completed!")
        print(f"Successfully evaluated: {len(self.all_results)} datasets")
        print(f"Failed evaluations: {len(self.failed_experiments)}")
        print(f"Results saved to: {self.results_dir}")
        
        return self.all_results
    
    def _save_results(self):
        """Save detailed results to JSON."""
        results_file = self.results_dir / 'robust_comparison_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        if self.failed_experiments:
            failures_file = self.results_dir / 'failed_experiments.json'
            with open(failures_file, 'w') as f:
                json.dump(self.failed_experiments, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
    
    def _create_summary(self):
        """Create summary statistics."""
        summary_data = []
        
        for dataset_result in self.all_results:
            dataset_name = dataset_result['dataset']
            for model_name, model_result in dataset_result['models'].items():
                if 'test_mse' in model_result and not np.isnan(model_result['test_mse']):
                    row = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'test_r2': model_result.get('test_r2', np.nan),
                        'test_mse': model_result['test_mse'],
                        'train_r2': model_result.get('train_r2', np.nan),
                        'train_mse': model_result.get('train_mse', np.nan),
                        'baseline_test_mse': model_result.get('baseline_test_mse', np.nan),
                        'test_mse_improvement': model_result.get('test_mse_improvement', np.nan),
                        'n_features': dataset_result['data_info']['n_features'],
                        'n_train_samples': dataset_result['data_info']['n_train_samples'],
                        'condition_number': dataset_result['data_info']['feature_matrix_condition']
                    }
                    
                    # Add KL divergence if valid
                    if 'kl_divergence' in model_result:
                        is_valid = model_result.get('is_valid_kl', True)
                        row['kl_divergence'] = model_result['kl_divergence'] if is_valid else np.nan
                        row['kl_valid'] = is_valid
                    else:
                        row['kl_divergence'] = np.nan
                        row['kl_valid'] = False
                    
                    # Add model-specific metrics
                    if model_name == 'ModelOrientedGeneralizedPCA':
                        row.update({
                            'r_beta_optimal': model_result.get('r_beta_optimal'),
                            'r_H_optimal': model_result.get('r_H_optimal'),
                            'bic_score': model_result.get('bic_score'),
                            'status': model_result.get('status', 'unknown')
                        })
                    elif model_name == 'PCA':
                        row.update({
                            'n_components': model_result.get('n_components'),
                            'variance_explained': model_result.get('cumulative_variance_explained')
                        })
                    
                    summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / 'robust_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Performance comparison by model (only successful runs)
        if len(summary_df) > 0:
            valid_df = summary_df[summary_df['test_mse'].notna()]
            
            perf_summary = valid_df.groupby('model').agg({
                'test_r2': ['mean', 'std', 'count'],
                'test_mse': ['mean', 'std', 'count'],
                'test_mse_improvement': ['mean', 'std'],
                'kl_divergence': ['mean', 'std', 'count']
            }).round(6)
            
            perf_file = self.results_dir / 'robust_performance_comparison.csv'
            perf_summary.to_csv(perf_file)
            
            print("\\nPerformance Summary (Robust Evaluation):")
            print(perf_summary)
            
            # Model selection summary for MOGPCA
            mogpca_df = valid_df[valid_df['model'] == 'ModelOrientedGeneralizedPCA']
            if len(mogpca_df) > 0:
                print("\\nMOGPCA Model Selection Summary:")
                selection_summary = mogpca_df.groupby(['r_beta_optimal', 'r_H_optimal']).size().reset_index(name='count')
                print(selection_summary)
        
        return summary_df
    
    def _create_visualizations(self):
        """Create robust comparison visualizations."""
        if not self.all_results:
            return
        
        # Collect data for plotting
        plot_data = []
        for dataset_result in self.all_results:
            dataset_name = dataset_result['dataset']
            for model_name, model_result in dataset_result['models'].items():
                if 'test_mse' in model_result and not np.isnan(model_result['test_mse']):
                    kl_div = model_result.get('kl_divergence', np.nan)
                    is_valid_kl = np.isfinite(kl_div) and kl_div >= 0
                    
                    plot_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'test_r2': model_result.get('test_r2', np.nan),
                        'test_mse': model_result['test_mse'],
                        'baseline_test_mse': model_result.get('baseline_test_mse', np.nan),
                        'test_mse_improvement': model_result.get('test_mse_improvement', np.nan),
                        'kl_divergence': kl_div if is_valid_kl else np.nan,
                        'kl_valid': is_valid_kl,
                        'cv_mse_mean': model_result.get('cv_mse_mean', np.nan),
                        'cv_mse_std': model_result.get('cv_mse_std', np.nan),
                        'cv_kl_mean': model_result.get('cv_kl_mean', np.nan),
                        'cv_kl_std': model_result.get('cv_kl_std', np.nan),
                        'n_successful_folds': model_result.get('n_successful_folds', 0)
                    })
        
        if not plot_data:
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create enhanced comparison plots with KL divergence and CV results
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Comparison: LR vs PCA vs Model-Oriented Generalized PCA', 
                     fontsize=16, fontweight='bold')
        
        models = df['model'].unique()
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        model_colors = dict(zip(models, colors[:len(models)]))
        
        # MSE comparison (primary metric)
        mse_data = [df[df['model'] == model]['test_mse'].values for model in models]
        bp1 = axes[0, 0].boxplot(mse_data, labels=models, patch_artist=True)
        for patch, model in zip(bp1['boxes'], models):
            patch.set_facecolor(model_colors[model])
        axes[0, 0].set_title('Test MSE Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Test MSE (log scale)')
        axes[0, 0].set_yscale('log')
        
        # MSE Improvement over baseline
        improvement_data = [df[df['model'] == model]['test_mse_improvement'].dropna().values for model in models]
        if any(len(data) > 0 for data in improvement_data):
            bp2 = axes[0, 1].boxplot([data for data in improvement_data if len(data) > 0], 
                                    labels=[model for model, data in zip(models, improvement_data) if len(data) > 0], 
                                    patch_artist=True)
            for patch, model in zip(bp2['boxes'], [model for model, data in zip(models, improvement_data) if len(data) > 0]):
                patch.set_facecolor(model_colors[model])
            axes[0, 1].set_title('MSE Improvement over Baseline')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylabel('MSE Improvement (higher is better)')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        else:
            axes[0, 1].text(0.5, 0.5, 'No improvement data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # KL divergence comparison (unified metric)
        valid_kl_df = df[df['kl_valid'] == True]
        if len(valid_kl_df) > 0:
            kl_models = valid_kl_df['model'].unique()
            kl_data = [valid_kl_df[valid_kl_df['model'] == model]['kl_divergence'].dropna().values for model in kl_models]
            if all(len(data) > 0 for data in kl_data):
                bp3 = axes[0, 2].boxplot(kl_data, labels=kl_models, patch_artist=True)
                for patch, model in zip(bp3['boxes'], kl_models):
                    patch.set_facecolor(model_colors[model])
                axes[0, 2].set_title('KL Divergence (Distributional Fit)')
                axes[0, 2].tick_params(axis='x', rotation=45)
                axes[0, 2].set_ylabel('KL Divergence (lower is better)')
                axes[0, 2].set_yscale('log')
            else:
                axes[0, 2].text(0.5, 0.5, 'No valid KL divergence data', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
        else:
            axes[0, 2].text(0.5, 0.5, 'No valid KL divergence data', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Cross-validation MSE comparison
        cv_mse_data = [df[df['model'] == model]['cv_mse_mean'].dropna().values for model in models]
        if any(len(data) > 0 for data in cv_mse_data):
            bp4 = axes[1, 0].boxplot([data for data in cv_mse_data if len(data) > 0], 
                                    labels=[model for model, data in zip(models, cv_mse_data) if len(data) > 0], 
                                    patch_artist=True)
            for patch, model in zip(bp4['boxes'], [model for model, data in zip(models, cv_mse_data) if len(data) > 0]):
                patch.set_facecolor(model_colors[model])
            axes[1, 0].set_title('Cross-Validation MSE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylabel('CV MSE (log scale)')
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].text(0.5, 0.5, 'No CV MSE data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Cross-validation KL divergence comparison
        cv_kl_data = [df[df['model'] == model]['cv_kl_mean'].dropna().values for model in models]
        if any(len(data) > 0 for data in cv_kl_data):
            bp5 = axes[1, 1].boxplot([data for data in cv_kl_data if len(data) > 0], 
                                    labels=[model for model, data in zip(models, cv_kl_data) if len(data) > 0], 
                                    patch_artist=True)
            for patch, model in zip(bp5['boxes'], [model for model, data in zip(models, cv_kl_data) if len(data) > 0]):
                patch.set_facecolor(model_colors[model])
            axes[1, 1].set_title('Cross-Validation KL Divergence')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylabel('CV KL Divergence (lower is better)')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No CV KL data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Summary scatter plot
        test_mse_vals = []
        kl_vals = []
        model_labels = []
        dataset_labels = []
        
        for _, row in df.iterrows():
            if row['kl_valid'] and not np.isnan(row['test_mse']):
                test_mse_vals.append(row['test_mse'])
                kl_vals.append(row['kl_divergence'])
                model_labels.append(row['model'])
                dataset_labels.append(row['dataset'])
        
        if test_mse_vals and kl_vals:
            for model in models:
                model_mask = [label == model for label in model_labels]
                if any(model_mask):
                    model_mse = [mse for mse, mask in zip(test_mse_vals, model_mask) if mask]
                    model_kl = [kl for kl, mask in zip(kl_vals, model_mask) if mask]
                    axes[1, 2].scatter(model_kl, model_mse, label=model, 
                                     color=model_colors[model], alpha=0.7, s=60)
            
            axes[1, 2].set_xlabel('KL Divergence')
            axes[1, 2].set_ylabel('Test MSE')
            axes[1, 2].set_title('KL Divergence vs Test MSE')
            axes[1, 2].set_xscale('log')
            axes[1, 2].set_yscale('log')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'No valid data for scatter plot', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig('enhanced_robust_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced comparison plots saved as 'enhanced_robust_comparison.png'")
    
    def run_enhanced_evaluation(self):
        """Run enhanced evaluation with K-fold CV and unified KL divergence."""
        datasets = ['abalone', 'bike_sharing_day', 'diabetes', 'energy_efficiency', 'wine_quality_red', 'concrete_strength']
        
        print("=" * 80)
        print("ENHANCED MODEL COMPARISON EVALUATION")
        print("Including K-Fold Cross-Validation and Unified KL Divergence")
        print("=" * 80)
        
        for dataset_name in datasets:
            try:
                result = self.evaluate_dataset(dataset_name)
                if result is not None:
                    self.all_results.append(result)
                    print(f"✓ Successfully evaluated {dataset_name}")
                else:
                    print(f"✗ Failed to evaluate {dataset_name}")
            except Exception as e:
                print(f"✗ Error evaluating {dataset_name}: {e}")
        
        if self.all_results:
            print("\nGenerating comprehensive analysis...")
            summary_df = self._create_summary()
            self._create_visualizations()
            
            # Save detailed results
            with open('enhanced_evaluation_results.json', 'w') as f:
                json.dump(self.all_results, f, indent=2, default=str)
            
            summary_df.to_csv('enhanced_evaluation_summary.csv', index=False)
            
            print("\nEvaluation complete! Results saved:")
            print("- enhanced_evaluation_results.json")
            print("- enhanced_evaluation_summary.csv") 
            print("- enhanced_robust_comparison.png")
        else:
            print("\nNo successful evaluations completed.")


if __name__ == "__main__":
    evaluator = RobustModelEvaluator(k_folds=5)
    evaluator.run_enhanced_evaluation()