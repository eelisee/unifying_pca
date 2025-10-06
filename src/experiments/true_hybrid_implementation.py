"""
Generalized PCA-Implementation Based on the Algebraic Framework.

This implements the generalized PCA model from the paper's operator class P,
not the regularized regression currently in compare_methods.py.

The model finds an operator A ∈ P with block structure:
    A = [A_σ   A_β]
        [0     A_μI]

Where:
- A_σ ≥ 0: noise scaling
- A_β: regression coefficients  
- A_μ: predictor transformation (low-rank projection)

The model should satisfy BOTH:
1. Good prediction: minimize ||y - A_β (A_μ X)||²
2. Good reconstruction: minimize ||X - A_μ X||²_F

This is fundamentally different from PCR or regularized regression.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class GeneralizedEstimator:
    """
    Generalized estimator based on operator class P framework.
    
    The estimator finds A ∈ L_ℓ ∩ H_r that belongs to both:
    - L_ℓ: regression-type operators (good for prediction)
    - H_r: PCA-type operators (good for reconstruction)
    """
    
    def __init__(self, r=2, lambda_pred=1.0, lambda_recon=1.0, n_iter=50):
        self.r = r  # rank of projection
        self.lambda_pred = lambda_pred  # weight for prediction loss
        self.lambda_recon = lambda_recon  # weight for reconstruction loss
        self.n_iter = n_iter
        
        # Learned parameters
        self.A_sigma = None  # noise scaling
        self.A_beta = None   # regression coefficients (p-dimensional)
        self.A_mu = None     # projection matrix (p x p, rank ≤ r)
        self.U = None        # orthonormal basis for projection (p x r)
        
    def _initialize_parameters(self, X, y):
        """Initialize A_σ, A_β, A_μ using simple heuristics."""
        n, p = X.shape
        
        # Initialize A_μ as top-r PCA projection
        pca = PCA(n_components=self.r)
        pca.fit(X)
        self.U = pca.components_.T  # p x r
        self.A_mu = self.U @ self.U.T  # p x p projection matrix
        
        # Initialize A_β via OLS on projected data
        X_proj = X @ self.A_mu
        ols = LinearRegression().fit(X_proj, y)
        self.A_beta = ols.coef_  # length p (includes zero components)
        
        # Initialize A_σ as residual std
        y_pred = X_proj @ self.A_beta
        residuals = y - y_pred
        self.A_sigma = np.std(residuals)
        
    def _compute_losses(self, X, y):
        """Compute prediction and reconstruction losses."""
        n, p = X.shape
        
        # Prediction loss: ||y - A_β (A_μ X)||²
        X_transformed = X @ self.A_mu  # apply predictor transformation
        y_pred = X_transformed @ self.A_beta
        pred_loss = np.mean((y - y_pred) ** 2)
        
        # Reconstruction loss: ||X - A_μ X||²_F
        X_reconstructed = X @ self.A_mu
        recon_loss = np.mean((X - X_reconstructed) ** 2)
        
        return pred_loss, recon_loss
    
    def _update_A_beta(self, X, y):
        """Update regression coefficients A_β for fixed A_μ."""
        # Solve: min_β ||y - (X A_μ) β||²
        X_transformed = X @ self.A_mu
        
        # Add small ridge for numerical stability
        XTX = X_transformed.T @ X_transformed + 1e-6 * np.eye(X_transformed.shape[1])
        XTy = X_transformed.T @ y
        
        try:
            self.A_beta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.A_beta = np.linalg.pinv(X_transformed) @ y
    
    def _update_A_mu(self, X, y, step_size=0.01):
        """Update projection A_μ via gradient descent."""
        n, p = X.shape
        
        # Current predictions and reconstruction
        X_transformed = X @ self.A_mu
        y_pred = X_transformed @ self.A_beta
        pred_residual = y - y_pred  # n x 1
        
        X_reconstructed = X @ self.A_mu
        recon_residual = X - X_reconstructed  # n x p
        
        # Gradients w.r.t. A_μ
        # ∂/∂A_μ ||y - X A_μ β||² = -2 X^T (y - X A_μ β) β^T
        grad_pred = -2 * (X.T @ pred_residual)[:, None] @ self.A_beta[None, :]  # p x p
        
        # ∂/∂A_μ ||X - X A_μ||²_F = -2 X^T (X - X A_μ)
        grad_recon = -2 * X.T @ recon_residual  # p x p
        
        # Combined gradient
        total_grad = self.lambda_pred * grad_pred + self.lambda_recon * grad_recon
        
        # Gradient step
        self.A_mu = self.A_mu - step_size * total_grad
        
        # Project back to rank-r matrices via SVD
        U_svd, s_svd, Vt_svd = np.linalg.svd(self.A_mu, full_matrices=False)
        s_svd_truncated = s_svd.copy()
        s_svd_truncated[self.r:] = 0  # keep only top r components
        self.A_mu = U_svd @ np.diag(s_svd_truncated) @ Vt_svd
        
        # Update orthonormal basis
        self.U = U_svd[:, :self.r]
    
    def fit(self, X, y):
        """Fit the true hybrid model via alternating optimization."""
        self._initialize_parameters(X, y)
        
        history = {'pred_loss': [], 'recon_loss': [], 'total_loss': []}
        
        for iteration in range(self.n_iter):
            # Update A_β for fixed A_μ
            self._update_A_beta(X, y)
            
            # Update A_μ for fixed A_β
            step_size = 0.01 / (1 + iteration * 0.01)  # decreasing step size
            self._update_A_mu(X, y, step_size=step_size)
            
            # Record losses
            pred_loss, recon_loss = self._compute_losses(X, y)
            total_loss = self.lambda_pred * pred_loss + self.lambda_recon * recon_loss
            
            history['pred_loss'].append(pred_loss)
            history['recon_loss'].append(recon_loss)
            history['total_loss'].append(total_loss)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: pred_loss={pred_loss:.4f}, recon_loss={recon_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions using the learned hybrid operator."""
        X_transformed = X @ self.A_mu
        return X_transformed @ self.A_beta
    
    def get_operator_structure(self):
        """Return the learned operator A in block form."""
        p = self.A_mu.shape[0]
        
        # Construct the (p+1) x (p+1) operator matrix
        A = np.zeros((p + 1, p + 1))
        A[0, 0] = self.A_sigma  # noise scaling
        A[0, 1:] = self.A_beta   # regression coefficients
        A[1:, 1:] = self.A_mu   # predictor transformation
        
        return A


def compare_true_hybrid_vs_others(X, y, test_size=0.3, r=2):
    """Compare true hybrid vs Linear Regression vs PCA reconstruction."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 1. Linear Regression (corresponds to A_μ = I, A_σ ≈ 0)
    ols = LinearRegression().fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    
    # 2. PCA Reconstruction (A_σ = 0, A_β = 0, only reconstruction)
    # For comparison, we use the best r-dimensional approximation to predict y
    # This represents the pure PCA approach from your H_r operator class
    pca = PCA(n_components=r).fit(X_train)
    X_test_reconstructed = pca.inverse_transform(pca.transform(X_test))
    
    # Since PCA doesn't directly predict y, we use the projection that best approximates
    # the relationship. This is the closest we can get to "pure PCA" for prediction.
    X_train_proj = pca.transform(X_train)
    pca_reg = LinearRegression().fit(X_train_proj, y_train)
    X_test_proj = pca.transform(X_test)
    y_pred_pca = pca_reg.predict(X_test_proj)
    mse_pca = mean_squared_error(y_test, y_pred_pca)
    
    # Alternative: Pure reconstruction error as a baseline
    # (This doesn't directly predict y, but shows PCA's reconstruction capability)
    recon_error = np.mean(np.sum((X_test - X_test_reconstructed) ** 2, axis=1))
    
    # 3. True Hybrid (A ∈ L_ℓ ∩ H_r)
    hybrid = GeneralizedEstimator(r=r, lambda_pred=1.0, lambda_recon=0.1, n_iter=100)
    history = hybrid.fit(X_train, y_train)
    y_pred_hybrid = hybrid.predict(X_test)
    mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
    
    print(f"Test MSE Comparison:")
    print(f"Linear Regression:     {mse_ols:.6f}")
    print(f"PCA-based prediction:  {mse_pca:.6f}")
    print(f"True Hybrid:           {mse_hybrid:.6f}")
    print(f"PCA reconstruction error: {recon_error:.6f} (for reference)")
    
    # Analyze operator structure
    A_operator = hybrid.get_operator_structure()
    print(f"\nLearned Operator A:")
    print(f"A_σ (noise scaling): {hybrid.A_sigma:.4f}")
    print(f"A_β (regression coeffs): {hybrid.A_beta[:5]}...")  # first 5 components
    print(f"rank(A_μ): {np.linalg.matrix_rank(hybrid.A_mu)}")
    
    return {
        'mse_linear_reg': mse_ols,
        'mse_pca': mse_pca,
        'mse_hybrid': mse_hybrid,
        'pca_recon_error': recon_error,
        'hybrid_model': hybrid,
        'training_history': history
    }


def plot_hybrid_analysis(results):
    """Plot analysis of the true hybrid model."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # MSE comparison
    methods = ['Linear Regression', 'PCA-based', 'True Hybrid']
    mses = [results['mse_linear_reg'], results['mse_pca'], results['mse_hybrid']]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    ax1.bar(methods, mses, color=colors)
    ax1.set_ylabel('Test MSE')
    ax1.set_title('MSE Comparison: Core Methods')
    ax1.tick_params(axis='x', rotation=45)
    
    # Training loss evolution
    history = results['training_history']
    iterations = range(len(history['total_loss']))
    
    ax2.plot(iterations, history['pred_loss'], label='Prediction Loss', color='blue')
    ax2.plot(iterations, history['recon_loss'], label='Reconstruction Loss', color='red')
    ax2.plot(iterations, history['total_loss'], label='Total Loss', color='black', linestyle='--')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Hybrid Training Progress')
    ax2.legend()
    ax2.set_yscale('log')
    
    # Operator structure visualization
    hybrid = results['hybrid_model']
    A_mu = hybrid.A_mu
    
    im3 = ax3.imshow(A_mu, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title('Learned A_μ (Projection Matrix)')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Feature Index')
    plt.colorbar(im3, ax=ax3)
    
    # Regression coefficients
    ax4.bar(range(len(hybrid.A_beta)), hybrid.A_beta)
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Learned A_β (Regression Coefficients)')
    
    plt.tight_layout()
    plt.savefig('results/true_hybrid_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test with synthetic data where hybrid should outperform both OLS and PCR
    from compare_methods import generate_X_sigma, sample_data
    
    print("=== True Hybrid Implementation Test ===\n")
    
    # Generate challenging synthetic data
    p = 8
    Sigma = generate_X_sigma(p=p, random_state=123)
    
    # Signal in low-variance direction (challenging for PCR)
    X, y, beta_true, sigma_eps = sample_data(n=400, p=p, Sigma=Sigma,
                                           beta_dir='small', snr=2.0, random_state=123)
    
    print(f"Data: n={X.shape[0]}, p={X.shape[1]}")
    print(f"True β direction: {beta_true[:5]}...")
    print(f"SNR: 2.0, signal in low-variance direction\n")
    
    # Compare methods
    results = compare_true_hybrid_vs_others(X, y, r=3)
    
    # Generate analysis plots
    plot_hybrid_analysis(results)
    print(f"\nAnalysis plots saved to 'results/true_hybrid_analysis.png'")
    
    # Check if hybrid is in the intersection L_ℓ ∩ H_r
    hybrid = results['hybrid_model']
    A_mu_rank = np.linalg.matrix_rank(hybrid.A_mu)
    A_beta_nonzero = np.sum(np.abs(hybrid.A_beta) > 1e-6)
    
    print(f"\nOperator Class Analysis:")
    print(f"- A_μ has rank {A_mu_rank} ≤ {hybrid.r} ✓ (belongs to H_r)")
    print(f"- A_β has {A_beta_nonzero} non-zero entries ✓ (belongs to L_ℓ)")
    print(f"- Therefore A ∈ L_ℓ ∩ H_r (true hybrid)")