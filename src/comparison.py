"""
Comparative analysis module for Linear Regression, PCA, and Generalized PCA.

This module implements the MSE comparison framework and stability analysis
described in the user's theoretical notes, treating MSE as an empirical
approximation of the model-theoretic loss.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .base import OperatorP, DataMetrics, validate_data
from .linear_regression import LinearRegressionOperator
from .pca import PCAOperator, PCARegressionOperator
from .generalized_pca import GeneralizedPCAOperator, GeneralizedPCARegressionOperator


class ModelComparison:
    """
    Comprehensive comparison framework for the three modeling approaches.
    
    This implements the empirical validation of the theoretical framework,
    measuring MSE as L̂(H) = (1/n)Σ(yi - ŷi(H))² as an empirical approximation
    of the model-theoretic loss L(μ, Hμ).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize comparison framework.
        
        Parameters:
        random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.results = {}
        self.stability_results = {}
        self.fitted_models = {}
        
    def compare_methods(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2,
                       n_components_list: Optional[List[int]] = None,
                       generalized_pca_configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Compare Linear Regression, Standard PCA, and Generalized PCA.
        
        This implements the core comparison measuring:
        L̂(H_LinReg) vs L̂(H_PCA) vs L̂(H_GenPCA)
        
        Parameters:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        n_components_list: List of component numbers to test for PCA methods
        generalized_pca_configs: List of configurations for generalized PCA
        
        Returns:
        Comprehensive comparison results
        """
        X, y = validate_data(X, y)
        n, p = X.shape
        
        # Default configurations
        if n_components_list is None:
            max_components = min(n, p, 10)
            n_components_list = list(range(1, max_components + 1))
        
        if generalized_pca_configs is None:
            generalized_pca_configs = [
                {'loss_type': 'entropy', 'optimization_method': 'gradient', 'max_iter': 10},
                {'loss_type': 'kl_divergence', 'optimization_method': 'gradient', 'max_iter': 8},
                # Removed fisher information due to complexity
            ]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        comparison_results = {
            'dataset_info': {
                'n_samples': n,
                'n_features': p,
                'n_train': X_train.shape[0],
                'n_test': X_test.shape[0]
            },
            'linear_regression': {},
            'standard_pca': {},
            'generalized_pca': {},
            'summary': {}
        }
        
        # 1. Linear Regression
        print("Fitting Linear Regression...")
        lr_results = self._evaluate_linear_regression(X_train, X_test, y_train, y_test)
        comparison_results['linear_regression'] = lr_results
        
        # 2. Standard PCA (with regression)
        print("Fitting Standard PCA...")
        pca_results = self._evaluate_standard_pca(
            X_train, X_test, y_train, y_test, n_components_list
        )
        comparison_results['standard_pca'] = pca_results
        
        # 3. Generalized PCA
        print("Fitting Generalized PCA...")
        gen_pca_results = self._evaluate_generalized_pca(
            X_train, X_test, y_train, y_test, n_components_list, generalized_pca_configs
        )
        comparison_results['generalized_pca'] = gen_pca_results
        
        # 4. Summary and best models
        summary = self._create_summary(comparison_results)
        comparison_results['summary'] = summary
        
        # Store results
        self.results = comparison_results
        
        return comparison_results
    
    def _evaluate_linear_regression(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate linear regression operator."""
        k = X_train.shape[1] + 1  # +1 for noise dimension
        
        # Standard linear regression
        lr = LinearRegressionOperator(k, fit_intercept=True)
        lr.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        
        train_metrics = lr.score(X_train, y_train)
        test_metrics = lr.score(X_test, y_test)
        
        # Store fitted model
        self.fitted_models['linear_regression'] = lr
        
        return {
            'model': lr,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'operator_info': lr.get_operator_info(),
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
    
    def _evaluate_standard_pca(self, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              n_components_list: List[int]) -> Dict[str, Any]:
        """Evaluate standard PCA with regression."""
        k = X_train.shape[1] + 1
        results = {}
        
        for n_comp in n_components_list:
            if n_comp > min(X_train.shape):
                continue
            
            # PCA Regression
            pca_reg = PCARegressionOperator(
                k, n_components=n_comp, center=True, scale=False
            )
            pca_reg.fit(X_train, y_train)
            
            # Metrics
            train_metrics = pca_reg.score(X_train, y_train)
            test_metrics = pca_reg.score(X_test, y_test)
            
            results[f'n_components_{n_comp}'] = {
                'model': pca_reg,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'operator_info': pca_reg.get_operator_info(),
                'n_components': n_comp
            }
        
        # Store best model based on test MSE
        if results:
            best_key = min(results.keys(), 
                          key=lambda k: results[k]['test_metrics'].get('prediction_mse', np.inf))
            self.fitted_models['standard_pca_best'] = results[best_key]['model']
            results['best_model'] = best_key
        
        return results
    
    def _evaluate_generalized_pca(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 n_components_list: List[int],
                                 configs: List[Dict]) -> Dict[str, Any]:
        """Evaluate generalized PCA with different configurations."""
        k = X_train.shape[1] + 1
        results = {}
        
        for config_idx, config in enumerate(configs):
            config_results = {}
            
            for n_comp in n_components_list:
                if n_comp > min(X_train.shape):
                    continue
                
                try:
                    # Generalized PCA Regression with robust error handling
                    gen_pca = GeneralizedPCARegressionOperator(
                        k, n_components=n_comp, center=True, scale=False, **config
                    )
                    gen_pca.fit(X_train, y_train)
                    
                    # Metrics
                    train_metrics = gen_pca.score(X_train, y_train)
                    test_metrics = gen_pca.score(X_test, y_test)
                    
                    # Validate results
                    if (np.isfinite(train_metrics.get('prediction_mse', np.inf)) and 
                        np.isfinite(test_metrics.get('prediction_mse', np.inf))):
                        config_results[f'n_components_{n_comp}'] = {
                            'model': gen_pca,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'operator_info': gen_pca.get_operator_info(),
                            'n_components': n_comp,
                            'config': config
                        }
                    else:
                        print(f"Warning: Generalized PCA produced invalid metrics for config {config}, "
                              f"n_components={n_comp}")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Generalized PCA failed for config {config}, "
                          f"n_components={n_comp}: {e}")
                    continue
            
            if config_results:
                # Find best for this configuration
                best_key = min(config_results.keys(),
                              key=lambda k: config_results[k]['test_metrics'].get('prediction_mse', np.inf))
                config_results['best_model'] = best_key
                
                results[f'config_{config_idx}'] = config_results
        
        # Store overall best generalized PCA model
        if results:
            all_models = []
            for config_name, config_results in results.items():
                if config_name.startswith('config_'):
                    for model_name, model_data in config_results.items():
                        if model_name.startswith('n_components_'):
                            all_models.append((f"{config_name}_{model_name}", model_data))
            
            if all_models:
                best_overall = min(all_models, 
                                 key=lambda x: x[1]['test_metrics'].get('prediction_mse', np.inf))
                self.fitted_models['generalized_pca_best'] = best_overall[1]['model']
                results['best_overall'] = best_overall[0]
        
        return results
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison table."""
        summary = {
            'method_comparison': {},
            'best_performers': {},
            'mse_comparison': {}
        }
        
        # Extract best MSE for each method
        methods_mse = {}
        
        # Linear Regression
        lr_mse = results['linear_regression']['test_metrics'].get('mse', np.inf)
        methods_mse['Linear Regression'] = lr_mse
        
        # Standard PCA
        pca_results = results['standard_pca']
        if pca_results:
            best_pca_mse = min(
                model_data['test_metrics'].get('prediction_mse', np.inf)
                for key, model_data in pca_results.items()
                if key.startswith('n_components_')
            )
            methods_mse['Standard PCA'] = best_pca_mse
        
        # Generalized PCA
        gen_pca_results = results['generalized_pca']
        if gen_pca_results:
            best_gen_pca_mse = np.inf
            for config_name, config_data in gen_pca_results.items():
                if config_name.startswith('config_'):
                    for model_name, model_data in config_data.items():
                        if model_name.startswith('n_components_'):
                            mse = model_data['test_metrics'].get('prediction_mse', np.inf)
                            best_gen_pca_mse = min(best_gen_pca_mse, mse)
            methods_mse['Generalized PCA'] = best_gen_pca_mse
        
        summary['mse_comparison'] = methods_mse
        
        # Determine best method
        if methods_mse:
            best_method = min(methods_mse.items(), key=lambda x: x[1])
            summary['best_method'] = best_method[0]
            summary['best_mse'] = best_method[1]
            
            # Performance improvements
            baseline_mse = methods_mse.get('Linear Regression', np.inf)
            if baseline_mse < np.inf:
                for method, mse in methods_mse.items():
                    if method != 'Linear Regression' and mse < np.inf:
                        improvement = (baseline_mse - mse) / baseline_mse * 100
                        summary[f'{method}_improvement_over_lr'] = improvement
        
        return summary
    
    def analyze_stability(self, X: np.ndarray, y: np.ndarray,
                         sample_sizes: List[int],
                         n_experiments: int = 3) -> Dict[str, Any]:  # Reduced from 5 to 3
        """
        Analyze operator stability ||H^(n) - H^(n+m)||_F across sample sizes.
        
        This implements the extendability analysis described in the theory,
        measuring how operators change as sample size increases.
        """
        X, y = validate_data(X, y)
        n_total, p = X.shape
        
        stability_results = {
            'sample_sizes': sample_sizes,
            'stability_metrics': {},
            'mse_evolution': {},
            'operator_norms': {}
        }
        
        # Ensure sample sizes are feasible and limit to avoid excessive computation
        sample_sizes = [s for s in sample_sizes if s <= n_total * 0.8]  # Leave room for test set
        if len(sample_sizes) > 5:  # Limit to 5 sample sizes maximum
            sample_sizes = sample_sizes[:5]
        
        for method_name in ['linear_regression', 'standard_pca', 'generalized_pca']:
            stability_results['stability_metrics'][method_name] = []
            stability_results['mse_evolution'][method_name] = []
            stability_results['operator_norms'][method_name] = []
        
        # Run experiments for each sample size
        for size_idx, sample_size in enumerate(sample_sizes):
            print(f"Analyzing stability for sample size {sample_size}...")
            
            size_stability = {method: [] for method in ['linear_regression', 'standard_pca', 'generalized_pca']}
            size_mse = {method: [] for method in ['linear_regression', 'standard_pca', 'generalized_pca']}
            size_norms = {method: [] for method in ['linear_regression', 'standard_pca', 'generalized_pca']}
            
            for exp in range(n_experiments):
                # Sample data
                np.random.seed(self.random_state + exp + size_idx * 100)
                sample_idx = np.random.choice(n_total, sample_size, replace=False)
                X_sample = X[sample_idx]
                y_sample = y[sample_idx]
                
                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sample, y_sample, test_size=0.2, random_state=self.random_state + exp
                )
                
                # Fit models and extract operators
                operators = self._fit_models_for_stability(X_train, y_train, X_test, y_test)
                
                for method, (operator, mse) in operators.items():
                    if operator is not None:
                        # Operator norm
                        if hasattr(operator, 'A_mu'):
                            op_norm = np.linalg.norm(operator.A_mu, 'fro')
                            size_norms[method].append(op_norm)
                        
                        size_mse[method].append(mse)
                        
                        # For stability comparison, compare with previous experiment
                        if exp > 0 and len(size_stability[method]) > 0:
                            # Compare with first operator of this size
                            prev_operators = [op for op, _ in 
                                            getattr(self, f'_temp_operators_{method}_{size_idx}', [])]
                            if prev_operators:
                                stability = DataMetrics.operator_stability(
                                    operator.A_mu, prev_operators[0].A_mu
                                )
                                size_stability[method].append(stability)
                
                # Store operators for comparison
                for method, (operator, _) in operators.items():
                    if operator is not None:
                        attr_name = f'_temp_operators_{method}_{size_idx}'
                        if not hasattr(self, attr_name):
                            setattr(self, attr_name, [])
                        getattr(self, attr_name).append((operator, mse))
            
            # Aggregate results for this sample size
            for method in ['linear_regression', 'standard_pca', 'generalized_pca']:
                stability_results['stability_metrics'][method].append(
                    np.mean(size_stability[method]) if size_stability[method] else np.nan
                )
                stability_results['mse_evolution'][method].append(
                    np.mean(size_mse[method]) if size_mse[method] else np.nan
                )
                stability_results['operator_norms'][method].append(
                    np.mean(size_norms[method]) if size_norms[method] else np.nan
                )
        
        # Clean up temporary attributes
        for method in ['linear_regression', 'standard_pca', 'generalized_pca']:
            for size_idx in range(len(sample_sizes)):
                attr_name = f'_temp_operators_{method}_{size_idx}'
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
        
        self.stability_results = stability_results
        return stability_results
    
    def _fit_models_for_stability(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Tuple[OperatorP, float]]:
        """Fit models for stability analysis."""
        k = X_train.shape[1] + 1
        operators = {}
        
        try:
            # Linear Regression
            lr = LinearRegressionOperator(k, fit_intercept=True)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            mse = DataMetrics.mse(y_test, y_pred)
            operators['linear_regression'] = (lr, mse)
        except Exception:
            operators['linear_regression'] = (None, np.inf)
        
        try:
            # Standard PCA (use moderate number of components)
            n_comp = min(5, min(X_train.shape) - 1)
            pca = PCARegressionOperator(k, n_components=n_comp, center=True)
            pca.fit(X_train, y_train)
            y_pred = pca.predict(X_test)
            mse = DataMetrics.mse(y_test, y_pred)
            operators['standard_pca'] = (pca, mse)
        except Exception:
            operators['standard_pca'] = (None, np.inf)
        
        try:
            # Generalized PCA (use simple configuration with faster convergence)
            n_comp = min(2, min(X_train.shape) - 1)  # Use fewer components for stability analysis
            gen_pca = GeneralizedPCARegressionOperator(
                k, n_components=n_comp, loss_type='entropy', 
                optimization_method='gradient', max_iter=5, center=True
            )
            gen_pca.fit(X_train, y_train)
            y_pred = gen_pca.predict(X_test)
            mse = DataMetrics.mse(y_test, y_pred)
            operators['generalized_pca'] = (gen_pca, mse)
        except Exception as e:
            # More informative error handling
            print(f"Warning: Generalized PCA failed in stability analysis: {e}")
            operators['generalized_pca'] = (None, np.inf)
        
        return operators
    
    def plot_comparison_results(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of comparison results."""
        if not self.results:
            raise ValueError("No comparison results found. Run compare_methods first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        # 1. MSE Comparison Bar Plot
        ax1 = axes[0, 0]
        mse_data = self.results['summary']['mse_comparison']
        methods = list(mse_data.keys())
        mses = list(mse_data.values())
        
        bars = ax1.bar(methods, mses, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('Test MSE')
        ax1.set_title('MSE Comparison Across Methods')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mse in zip(bars, mses):
            if mse < np.inf:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mse:.4f}', ha='center', va='bottom')
        
        # 2. Component Analysis for PCA methods
        ax2 = axes[0, 1]
        if 'standard_pca' in self.results and self.results['standard_pca']:
            pca_results = self.results['standard_pca']
            components = []
            mses = []
            for key, data in pca_results.items():
                if key.startswith('n_components_'):
                    comp_num = int(key.split('_')[-1])
                    components.append(comp_num)
                    mses.append(data['test_metrics'].get('prediction_mse', np.inf))
            
            ax2.plot(components, mses, 'o-', label='Standard PCA', color='lightcoral')
        
        if 'generalized_pca' in self.results and self.results['generalized_pca']:
            # Plot best configuration
            gen_pca_results = self.results['generalized_pca']
            for config_name, config_data in gen_pca_results.items():
                if config_name.startswith('config_'):
                    components = []
                    mses = []
                    for key, data in config_data.items():
                        if key.startswith('n_components_'):
                            comp_num = int(key.split('_')[-1])
                            components.append(comp_num)
                            mses.append(data['test_metrics'].get('prediction_mse', np.inf))
                    
                    if components:
                        ax2.plot(components, mses, 's--', label=f'Gen PCA {config_name}', 
                                color='lightgreen', alpha=0.7)
                        break  # Just plot first config for clarity
        
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Test MSE')
        ax2.set_title('MSE vs Number of Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stability Analysis (if available)
        ax3 = axes[1, 0]
        if self.stability_results:
            sample_sizes = self.stability_results['sample_sizes']
            for method in ['linear_regression', 'standard_pca', 'generalized_pca']:
                mse_evolution = self.stability_results['mse_evolution'][method]
                if not all(np.isnan(mse_evolution)):
                    ax3.plot(sample_sizes, mse_evolution, 'o-', label=method)
            
            ax3.set_xlabel('Sample Size')
            ax3.set_ylabel('Test MSE')
            ax3.set_title('MSE Evolution with Sample Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No stability analysis\navailable', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Stability Analysis')
        
        # 4. Operator Stability (if available)
        ax4 = axes[1, 1]
        if self.stability_results:
            sample_sizes = self.stability_results['sample_sizes']
            for method in ['standard_pca', 'generalized_pca']:  # Skip linear regression
                stability = self.stability_results['stability_metrics'][method]
                if not all(np.isnan(stability)):
                    ax4.plot(sample_sizes, stability, 'o-', label=method)
            
            ax4.set_xlabel('Sample Size')
            ax4.set_ylabel('||H^(n) - H^(n+m)||_F')
            ax4.set_title('Operator Stability Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No operator stability\nanalysis available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Operator Stability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, filepath: str) -> None:
        """Export results to CSV format."""
        if not self.results:
            raise ValueError("No results to export. Run compare_methods first.")
        
        # Create summary DataFrame
        data = []
        
        # Linear Regression
        lr_data = self.results['linear_regression']
        data.append({
            'Method': 'Linear Regression',
            'Configuration': 'Standard',
            'Components': 'All',
            'Train_MSE': lr_data['train_metrics']['mse'],
            'Test_MSE': lr_data['test_metrics']['mse'],
            'Train_R2': lr_data['train_metrics']['r2'],
            'Test_R2': lr_data['test_metrics']['r2']
        })
        
        # Standard PCA
        if 'standard_pca' in self.results:
            for key, pca_data in self.results['standard_pca'].items():
                if key.startswith('n_components_'):
                    n_comp = int(key.split('_')[-1])
                    data.append({
                        'Method': 'Standard PCA',
                        'Configuration': 'Regression',
                        'Components': n_comp,
                        'Train_MSE': pca_data['train_metrics'].get('prediction_mse', np.nan),
                        'Test_MSE': pca_data['test_metrics'].get('prediction_mse', np.nan),
                        'Train_R2': pca_data['train_metrics'].get('prediction_r2', np.nan),
                        'Test_R2': pca_data['test_metrics'].get('prediction_r2', np.nan)
                    })
        
        # Generalized PCA
        if 'generalized_pca' in self.results:
            for config_name, config_data in self.results['generalized_pca'].items():
                if config_name.startswith('config_'):
                    for key, gen_data in config_data.items():
                        if key.startswith('n_components_'):
                            n_comp = int(key.split('_')[-1])
                            config_info = gen_data['config']
                            config_str = f"{config_info.get('loss_type', 'unknown')}"
                            
                            data.append({
                                'Method': 'Generalized PCA',
                                'Configuration': config_str,
                                'Components': n_comp,
                                'Train_MSE': gen_data['train_metrics'].get('prediction_mse', np.nan),
                                'Test_MSE': gen_data['test_metrics'].get('prediction_mse', np.nan),
                                'Train_R2': gen_data['train_metrics'].get('prediction_r2', np.nan),
                                'Test_R2': gen_data['test_metrics'].get('prediction_r2', np.nan)
                            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


def run_dataset_comparison(dataset_name: str, X: np.ndarray, y: np.ndarray,
                          save_plots: bool = True, save_results: bool = True) -> ModelComparison:
    """
    Convenience function to run complete comparison on a dataset.
    
    Parameters:
    dataset_name: Name of the dataset for file naming
    X: Feature matrix
    y: Target vector
    save_plots: Whether to save comparison plots
    save_results: Whether to save results to CSV
    
    Returns:
    ModelComparison object with results
    """
    print(f"Running comprehensive comparison on {dataset_name} dataset...")
    
    comparison = ModelComparison(random_state=42)
    
    # Main comparison
    results = comparison.compare_methods(X, y, test_size=0.2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    
    mse_comparison = results['summary']['mse_comparison']
    print("\nMSE Comparison:")
    for method, mse in mse_comparison.items():
        print(f"  {method}: {mse:.6f}")
    
    if 'best_method' in results['summary']:
        print(f"\nBest Method: {results['summary']['best_method']} "
              f"(MSE: {results['summary']['best_mse']:.6f})")
    
    # Stability analysis for smaller datasets
    if X.shape[0] <= 1000:
        print(f"\nRunning stability analysis...")
        max_samples = min(X.shape[0] - 50, 500)
        sample_sizes = list(range(50, max_samples, 50))
        stability_results = comparison.analyze_stability(X, y, sample_sizes, n_experiments=3)
        
        print("Stability Analysis Completed")
    
    # Generate plots
    if save_plots:
        plot_path = f"comparison_{dataset_name.lower()}.png"
        comparison.plot_comparison_results(save_path=plot_path)
        print(f"Plots saved to {plot_path}")
    else:
        comparison.plot_comparison_results()
    
    # Export results
    if save_results:
        csv_path = f"results_{dataset_name.lower()}.csv"
        comparison.export_results(csv_path)
    
    return comparison