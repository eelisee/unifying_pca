"""
Model-Oriented Generalized PCA Implementation

This module implements a model-oriented algorithm to estimate an operator A ∈ P
for a given empirical dataset (X,Y). The operator A integrates a regression 
component and a PCA-like projection and is chosen by minimizing a model-level 
loss (KL divergence) between the empirical model and the model induced by A.

References:
- Based on the mathematical framework described in the implementation specification
- Uses alternating subspace optimization with KL divergence minimization
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from typing import Tuple, Optional, Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOrientedGeneralizedPCA:
    """
    Model-Oriented Generalized PCA with alternating subspace optimization.
    
    This class implements a generalized PCA that combines regression and 
    dimensionality reduction by minimizing KL divergence between empirical
    and model-induced Gaussian distributions.
    
    Parameters
    ----------
    r_beta_candidates : list of int, optional
        Candidate values for regression subspace dimension (default: [0, 1, 2, 3, 4, 5])
    r_H_candidates : list of int, optional  
        Candidate values for PCA subspace dimension (default: [1, 2, 3, 4, 5, 6, 7, 8])
    max_iter : int, optional
        Maximum number of alternating iterations (default: 100)
    tol : float, optional
        Convergence tolerance for KL divergence decrease (default: 1e-6)
    regularization : float, optional
        Regularization parameter for numerical stability (default: 1e-8)
    use_pls : bool, optional
        Whether to use PLS for V_beta updates (default: True for high-dim data)
    model_selection : str, optional
        Model selection criterion: 'aic', 'bic', or 'cv' (default: 'bic')
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, 
                 r_beta_candidates: Optional[List[int]] = None,
                 r_H_candidates: Optional[List[int]] = None,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 regularization: float = 1e-8,
                 use_pls: Optional[bool] = None,
                 model_selection: str = 'bic',
                 random_state: int = 42):
        
        self.r_beta_candidates = r_beta_candidates or [0, 1, 2, 3, 4, 5]
        self.r_H_candidates = r_H_candidates or [1, 2, 3, 4, 5, 6, 7, 8]
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.use_pls = use_pls
        self.model_selection = model_selection
        self.random_state = random_state
        
        # Results storage
        self.A_sigma_ = None
        self.A_beta_ = None  
        self.A_mu_ = None
        self.V_beta_ = None
        self.V_H_ = None
        self.r_beta_optimal_ = None
        self.r_H_optimal_ = None
        self.kl_divergence_ = None
        self.aic_score_ = None
        self.bic_score_ = None
        self.n_iter_ = None
        self.Sigma_Z_empirical_ = None
        self.Sigma_Z_model_ = None
        
        # Store fitted components
        self.X_mean_ = None
        self.Y_mean_ = None
        self.beta_ols_ = None
        self.residual_scale_ = None  # Changed from residual_variance_ to residual_scale_
        
        np.random.seed(random_state)
    
    def _center_data(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Center the data matrices."""
        self.X_mean_ = np.mean(X, axis=0)
        X_centered = X - self.X_mean_
        
        Y_centered = None
        if Y is not None:
            self.Y_mean_ = np.mean(Y, axis=0)
            Y_centered = Y - self.Y_mean_
        
        return X_centered, Y_centered
    
    def _compute_ols_baseline(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute OLS baseline and residual scale (standard deviation, not variance)."""
        n, p = X.shape
        
        # Handle singular X^T X case
        try:
            if p > n or np.linalg.cond(X.T @ X) > 1e12:
                # Use pseudoinverse for singular case
                beta_ols = la.pinv(X) @ Y
            else:
                beta_ols = la.solve(X.T @ X, X.T @ Y)
        except la.LinAlgError:
            beta_ols = la.pinv(X) @ Y
        
        # Compute residuals and variance
        residuals = Y - X @ beta_ols
        if Y.ndim == 1:
            residual_var = np.sum(residuals**2) / max(1, n - p)
        else:
            residual_var = np.trace(residuals.T @ residuals) / max(1, n - p)
        
        return beta_ols, np.sqrt(residual_var)  # Return scale (std dev), not variance
    
    def _build_empirical_covariance(self, X: np.ndarray) -> np.ndarray:
        """Build empirical covariance matrix Σ_Z."""
        n, p = X.shape
        
        # Empirical covariance of X
        Sigma_X = (X.T @ X) / n
        
        # Build augmented covariance (with noise component)
        Sigma_Z = np.zeros((1 + p, 1 + p))
        Sigma_Z[0, 0] = 1.0  # Variance of ε ~ N(0,1)
        Sigma_Z[1:, 1:] = Sigma_X
        # Cross-covariances with ε set to zero (latent)
        
        return Sigma_Z
    
    def _build_operator(self, A_sigma: float, A_beta: np.ndarray, V_H: np.ndarray) -> np.ndarray:
        """Build the block operator A."""
        p = V_H.shape[0]
        A_mu = V_H @ V_H.T
        
        A = np.zeros((1 + p, 1 + p))
        A[0, 0] = A_sigma
        A[0, 1:] = A_beta.flatten()
        A[1:, 1:] = A_mu
        
        return A
    
    def _compute_model_covariance(self, A: np.ndarray, Sigma_Z_emp: np.ndarray) -> np.ndarray:
        """Compute model-implied covariance Σ'_Z(A) = A Σ_Z A^T."""
        return A @ Sigma_Z_emp @ A.T
    
    def _kl_divergence_gaussians(self, Sigma_emp: np.ndarray, Sigma_model: np.ndarray) -> float:
        """Compute KL divergence between two multivariate Gaussians."""
        m = Sigma_emp.shape[0]
        
        # Add regularization for numerical stability
        Sigma_model_reg = Sigma_model + self.regularization * np.eye(m)
        
        try:
            # Compute inverse and determinants
            Sigma_model_inv = la.inv(Sigma_model_reg)
            
            # Use slogdet for numerical stability
            sign_emp, logdet_emp = np.linalg.slogdet(Sigma_emp)
            sign_model, logdet_model = np.linalg.slogdet(Sigma_model_reg)
            
            if sign_emp <= 0 or sign_model <= 0:
                logger.warning("Non-positive definite covariance matrix detected")
                return np.inf
            
            # KL divergence formula
            trace_term = np.trace(Sigma_model_inv @ Sigma_emp)
            det_term = logdet_model - logdet_emp
            kl = 0.5 * (trace_term - m + det_term)
            
            return kl
            
        except (la.LinAlgError, np.linalg.LinAlgError):
            logger.warning("Numerical issues in KL divergence computation")
            return np.inf
    
    def _negative_log_likelihood(self, Sigma_Z_emp: np.ndarray, Sigma_Z_model: np.ndarray, n: int) -> float:
        """Compute negative log-likelihood of empirical data under model."""
        m = Sigma_Z_emp.shape[0]
        
        # Add regularization
        Sigma_model_reg = Sigma_Z_model + self.regularization * np.eye(m)
        
        try:
            Sigma_model_inv = la.inv(Sigma_model_reg)
            sign, logdet = np.linalg.slogdet(Sigma_model_reg)
            
            if sign <= 0:
                return np.inf
            
            nll = (n / 2) * (logdet + np.trace(Sigma_Z_emp @ Sigma_model_inv))
            return nll
            
        except (la.LinAlgError, np.linalg.LinAlgError):
            return np.inf
    
    def _count_parameters(self, r_beta: int, r_H: int, p: int) -> int:
        """Count effective number of parameters."""
        # A_sigma: 1 parameter
        # A_beta: r_beta * p parameters (regression coefficients in subspace)
        # V_H: r_H * p - r_H*(r_H-1)/2 parameters (orthogonal matrix on Stiefel manifold)
        k = 1 + r_beta * p + r_H * p - r_H * (r_H - 1) // 2
        return k
    
    def _update_V_beta_ols(self, X: np.ndarray, Y: np.ndarray, r_beta: int) -> np.ndarray:
        """Update V_beta using OLS-based approach."""
        if r_beta == 0:
            return np.zeros((X.shape[1], 0))
        
        # Recompute OLS coefficients
        beta_ols, _ = self._compute_ols_baseline(X, Y)
        
        if Y.ndim == 1:
            # Univariate Y: take top directions from beta
            if r_beta == 1:
                beta_norm = beta_ols / (np.linalg.norm(beta_ols) + 1e-10)
                return beta_norm.reshape(-1, 1)
            else:
                # For r_beta > 1, use random orthogonal directions (could be improved)
                p = X.shape[1]
                V_beta = np.random.randn(p, r_beta)
                V_beta[:, 0] = beta_ols / (np.linalg.norm(beta_ols) + 1e-10)
                V_beta, _ = la.qr(V_beta)
                return V_beta
        else:
            # Multivariate Y: use SVD of beta matrix
            U, s, Vt = la.svd(beta_ols, full_matrices=False)
            r_eff = min(r_beta, len(s))
            return U[:, :r_eff]
    
    def _update_V_beta_pls(self, X: np.ndarray, Y: np.ndarray, r_beta: int) -> np.ndarray:
        """
        Update V_beta using PLS approach.
        
        Note: This is a numerical shortcut, not part of the core theoretical model.
        PLS finds directions in X that have maximal covariance with Y.
        """
        if r_beta == 0:
            return np.zeros((X.shape[1], 0))
        
        # Use PLS to find predictive directions
        n_components = min(r_beta, min(X.shape[1], X.shape[0] - 1))
        
        try:
            if Y.ndim == 1:
                Y_pls = Y.reshape(-1, 1)
            else:
                Y_pls = Y
            
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X, Y_pls)
            
            # Extract loading directions (X-space components)
            V_beta = pls.x_loadings_
            
            # Orthogonalize
            V_beta, _ = la.qr(V_beta)
            
            return V_beta
            
        except Exception as e:
            logger.warning(f"PLS failed, falling back to OLS: {e}")
            return self._update_V_beta_ols(X, Y, r_beta)
    
    def _update_V_H(self, Sigma_X: np.ndarray, V_beta: np.ndarray, r_H: int) -> np.ndarray:
        """Update V_H by eigen-decomposition of residual covariance."""
        p = Sigma_X.shape[0]
        
        # Compute projection onto V_beta complement
        if V_beta.shape[1] > 0:
            P_V_beta = V_beta @ V_beta.T
            I_minus_P = np.eye(p) - P_V_beta
        else:
            I_minus_P = np.eye(p)
        
        # Residual covariance
        Sigma_R = I_minus_P @ Sigma_X @ I_minus_P
        
        # Add small regularization for numerical stability
        Sigma_R += self.regularization * np.eye(p)
        
        # Compute top r_H eigenvectors
        try:
            if r_H >= p or p <= 50:  # Use dense solver for small problems
                eigenvals, eigenvecs = la.eigh(Sigma_R)
                # Sort in descending order
                idx = np.argsort(eigenvals)[::-1]
                V_H = eigenvecs[:, idx[:r_H]]
            else:  # Use sparse solver for large problems
                eigenvals, eigenvecs = spla.eigsh(Sigma_R, k=r_H, which='LA')
                # eigsh returns in ascending order, reverse
                idx = np.argsort(eigenvals)[::-1]
                V_H = eigenvecs[:, idx]
            
            return V_H
            
        except Exception as e:
            logger.warning(f"Eigendecomposition failed, using random initialization: {e}")
            V_H = np.random.randn(p, r_H)
            V_H, _ = la.qr(V_H)
            return V_H
    
    def _fit_single_configuration(self, X: np.ndarray, Y: np.ndarray, r_beta: int, r_H: int) -> Dict[str, Any]:
        """Fit model for a single (r_beta, r_H) configuration."""
        n, p = X.shape
        
        # Determine whether to use PLS
        use_pls = self.use_pls
        if use_pls is None:
            use_pls = (p >= n) or (p > 50)  # Use PLS for high-dimensional or underdetermined case
        
        # Initialize V_beta
        if use_pls:
            V_beta = self._update_V_beta_pls(X, Y, r_beta)
        else:
            V_beta = self._update_V_beta_ols(X, Y, r_beta)
        
        # Empirical covariances
        Sigma_Z_emp = self._build_empirical_covariance(X)
        Sigma_X = Sigma_Z_emp[1:, 1:]
        
        # Initialize A_sigma and A_beta
        A_sigma = self.residual_scale_  # A_σ is the scale component (std dev), not variance
        if r_beta > 0:
            # Fix: Project beta onto V_beta subspace correctly: A_β = P_{V_β} β̂
            A_beta = (V_beta @ V_beta.T) @ self.beta_ols_
            if A_beta.ndim == 2:
                A_beta = A_beta.flatten()
        else:
            A_beta = np.zeros(p)
        
        kl_history = []
        
        # Alternating optimization
        for iteration in range(self.max_iter):
            # Update V_H (spectral step)
            V_H = self._update_V_H(Sigma_X, V_beta, r_H)
            
            # Refit beta on projected predictor X_proj = X @ V_H @ V_H.T
            if iteration > 0:  # Skip first iteration to avoid double computation
                X_proj = X @ (V_H @ V_H.T)
                beta_refit, _ = self._compute_ols_baseline(X_proj, Y)
                self.beta_ols_ = beta_refit  # Update for next iteration
            
            # Build operator and compute KL divergence
            A = self._build_operator(A_sigma, A_beta, V_H)
            Sigma_Z_model = self._compute_model_covariance(A, Sigma_Z_emp)
            kl = self._kl_divergence_gaussians(Sigma_Z_emp, Sigma_Z_model)
            
            kl_history.append(kl)
            
            # Check convergence
            if iteration > 0 and abs(kl_history[-2] - kl_history[-1]) < self.tol:
                break
            
            # Update V_beta (predictive step)
            if use_pls:
                V_beta_new = self._update_V_beta_pls(X, Y, r_beta)
            else:
                V_beta_new = self._update_V_beta_ols(X, Y, r_beta)
            
            V_beta = V_beta_new
            
            # Update A_beta with corrected projection
            if r_beta > 0:
                A_beta = (V_beta @ V_beta.T) @ self.beta_ols_
                if A_beta.ndim == 2:
                    A_beta = A_beta.flatten()
        
        # Final operator and metrics
        A_final = self._build_operator(A_sigma, A_beta, V_H)
        Sigma_Z_model_final = self._compute_model_covariance(A_final, Sigma_Z_emp)
        kl_final = self._kl_divergence_gaussians(Sigma_Z_emp, Sigma_Z_model_final)
        
        # Model selection scores
        nll = self._negative_log_likelihood(Sigma_Z_emp, Sigma_Z_model_final, n)
        k = self._count_parameters(r_beta, r_H, p)
        # Fix: Correct AIC/BIC calculation
        aic = 2 * k + 2 * nll
        bic = np.log(n) * k + 2 * nll
        
        return {
            'V_beta': V_beta,
            'V_H': V_H,
            'A_sigma': A_sigma,
            'A_beta': A_beta,
            'A': A_final,
            'kl_divergence': kl_final,
            'aic': aic,
            'bic': bic,
            'nll': nll,
            'n_iter': iteration + 1,
            'kl_history': kl_history,
            'Sigma_Z_empirical': Sigma_Z_emp,
            'Sigma_Z_model': Sigma_Z_model_final
        }
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'ModelOrientedGeneralizedPCA':
        """
        Fit the model-oriented generalized PCA.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Training data targets
            
        Returns
        -------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n, p = X.shape
        
        # Step 1: Center data
        X_centered, Y_centered = self._center_data(X, Y)
        
        # Step 2: Compute OLS baseline
        self.beta_ols_, self.residual_scale_ = self._compute_ols_baseline(X_centered, Y_centered)
        
        # Step 3: Adjust candidate ranges based on data dimensions
        max_r_beta = min(max(self.r_beta_candidates), p, n-1)
        max_r_H = min(max(self.r_H_candidates), p)
        
        r_beta_candidates = [r for r in self.r_beta_candidates if r <= max_r_beta]
        r_H_candidates = [r for r in self.r_H_candidates if r <= max_r_H]
        
        logger.info(f"Fitting model with data shape {X.shape}, "
                   f"r_beta candidates: {r_beta_candidates}, "
                   f"r_H candidates: {r_H_candidates}")
        
        # Step 4: Grid search over (r_beta, r_H) pairs
        best_score = np.inf
        best_config = None
        results = {}
        
        for r_beta in r_beta_candidates:
            for r_H in r_H_candidates:
                try:
                    config_result = self._fit_single_configuration(X_centered, Y_centered, r_beta, r_H)
                    results[(r_beta, r_H)] = config_result
                    
                    # Model selection
                    if self.model_selection == 'aic':
                        score = config_result['aic']
                    elif self.model_selection == 'bic':
                        score = config_result['bic']
                    else:  # kl
                        score = config_result['kl_divergence']
                    
                    if score < best_score:
                        best_score = score
                        best_config = (r_beta, r_H)
                        
                    logger.info(f"r_beta={r_beta}, r_H={r_H}: KL={config_result['kl_divergence']:.6f}, "
                               f"AIC={config_result['aic']:.2f}, BIC={config_result['bic']:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fit r_beta={r_beta}, r_H={r_H}: {e}")
                    continue
        
        if best_config is None:
            raise RuntimeError("No valid configuration found during grid search")
        
        # Step 5: Store optimal results
        best_result = results[best_config]
        self.r_beta_optimal_ = best_config[0]
        self.r_H_optimal_ = best_config[1]
        self.V_beta_ = best_result['V_beta']
        self.V_H_ = best_result['V_H']
        self.A_sigma_ = best_result['A_sigma']
        self.A_beta_ = best_result['A_beta']
        self.A_mu_ = self.V_H_ @ self.V_H_.T
        self.kl_divergence_ = best_result['kl_divergence']
        self.aic_score_ = best_result['aic']
        self.bic_score_ = best_result['bic']
        self.n_iter_ = best_result['n_iter']
        self.Sigma_Z_empirical_ = best_result['Sigma_Z_empirical']
        self.Sigma_Z_model_ = best_result['Sigma_Z_model']
        
        logger.info(f"Optimal configuration: r_beta={self.r_beta_optimal_}, r_H={self.r_H_optimal_}")
        logger.info(f"Final KL divergence: {self.kl_divergence_:.6f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using the fitted PCA subspace V_H.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        X_transformed : array of shape (n_samples, r_H_optimal_)
            Transformed data
        """
        if self.V_H_ is None:
            raise RuntimeError("Model must be fitted before transform")
        
        X = np.asarray(X)
        X_centered = X - self.X_mean_
        return X_centered @ self.V_H_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        if self.V_beta_ is None or self.beta_ols_ is None:
            raise RuntimeError("Model must be fitted before predict")
        
        X = np.asarray(X)
        X_centered = X - self.X_mean_
        
        # Project onto regression subspace and predict
        if self.r_beta_optimal_ > 0:
            X_proj = X_centered @ self.V_beta_ @ self.V_beta_.T
            y_pred = X_proj @ self.beta_ols_
        else:
            # No regression subspace, predict mean
            if hasattr(self, 'Y_mean_') and self.Y_mean_ is not None:
                y_pred = np.full((X.shape[0], len(self.Y_mean_)), self.Y_mean_)
            else:
                y_pred = np.zeros((X.shape[0], 1))
        
        # Add back mean
        if hasattr(self, 'Y_mean_') and self.Y_mean_ is not None:
            y_pred += self.Y_mean_
        
        return y_pred.squeeze()
    
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. Y
        """
        y_pred = self.predict(X)
        return r2_score(Y, y_pred)
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get the fitted components.
        
        Returns
        -------
        components : dict
            Dictionary containing fitted components:
            - 'V_beta': regression subspace basis
            - 'V_H': PCA subspace basis  
            - 'A_mu': structural operator (projection matrix)
            - 'beta_ols': OLS coefficients
        """
        if self.V_H_ is None:
            raise RuntimeError("Model must be fitted before accessing components")
        
        return {
            'V_beta': self.V_beta_,
            'V_H': self.V_H_,
            'A_mu': self.A_mu_,
            'beta_ols': self.beta_ols_,
            'A_sigma': self.A_sigma_,
            'A_beta': self.A_beta_
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the fitted model.
        
        Returns
        -------
        diagnostics : dict
            Dictionary containing diagnostic information
        """
        if self.kl_divergence_ is None:
            raise RuntimeError("Model must be fitted before accessing diagnostics")
        
        return {
            'kl_divergence': self.kl_divergence_,
            'aic_score': self.aic_score_,
            'bic_score': self.bic_score_,
            'r_beta_optimal': self.r_beta_optimal_,
            'r_H_optimal': self.r_H_optimal_,
            'n_iterations': self.n_iter_,
            'residual_scale': self.residual_scale_,  # Updated name
            'model_selection_criterion': self.model_selection
        }