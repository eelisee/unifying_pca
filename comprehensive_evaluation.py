"""
Comprehensive evaluation script for all methods on all datasets.

This script runs Linear Regression, Standard PCA, and Generalized PCA on all 
preprocessed datasets and generates complete results with visualizations.
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

from src.comparison import ModelComparison
from src.linear_regression import LinearRegressionOperator
from src.pca import PCARegressionOperator
from src.generalized_pca import GeneralizedPCARegressionOperator
from src.base import DataMetrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ComprehensiveEvaluation:
    """
    Comprehensive evaluation framework for all methods on all datasets.
    """
    
    def __init__(self, data_dir: str = "data/processed", results_dir: str = "results"):
        """
        Initialize the evaluation framework.
        
        Parameters:
        data_dir: Directory containing processed datasets
        results_dir: Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.all_results = {}
        self.summary_results = []
        self.failed_experiments = []
        
        # Available datasets
        self.datasets = self._discover_datasets()
        
        print(f"Found {len(self.datasets)} datasets: {list(self.datasets.keys())}")
    
    def _discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available processed datasets."""
        datasets = {}
        
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != "__pycache__":
                metadata_file = dataset_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    datasets[dataset_dir.name] = {
                        'path': dataset_dir,
                        'metadata': metadata
                    }
        
        return datasets
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a processed dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset_path = self.datasets[dataset_name]['path']
        
        # Load data files
        X_train = pd.read_csv(dataset_path / "X_train_scaled.csv").values
        X_test = pd.read_csv(dataset_path / "X_test_scaled.csv").values
        y_train = pd.read_csv(dataset_path / "y_train.csv").iloc[:, 0].values  # First column
        y_test = pd.read_csv(dataset_path / "y_test.csv").iloc[:, 0].values    # First column
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate all methods on a single dataset.
        
        Parameters:
        dataset_name: Name of the dataset to evaluate
        
        Returns:
        Complete evaluation results for the dataset
        """
        print(f"\\n{'='*60}")
        print(f"EVALUATING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_dataset(dataset_name)
            metadata = self.datasets[dataset_name]['metadata']
            
            print(f"Dataset shape: {X_train.shape[0] + X_test.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}")
            
            # Combine for full dataset analysis if needed
            X_full = np.vstack([X_train, X_test])
            y_full = np.hstack([y_train, y_test])
            
            # Initialize results
            dataset_results = {
                'dataset_name': dataset_name,
                'metadata': metadata,
                'data_shape': {
                    'n_samples_train': X_train.shape[0],
                    'n_samples_test': X_test.shape[0],
                    'n_features': X_train.shape[1]
                },
                'methods': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. Linear Regression
            print("\\nEvaluating Linear Regression...")
            lr_results = self._evaluate_linear_regression(X_train, X_test, y_train, y_test)
            dataset_results['methods']['linear_regression'] = lr_results
            
            # 2. Standard PCA Regression
            print("\\nEvaluating Standard PCA Regression...")
            pca_results = self._evaluate_pca_regression(X_train, X_test, y_train, y_test)
            dataset_results['methods']['pca_regression'] = pca_results
            
            # 3. Generalized PCA (Theoretical Implementation)
            print("\\nEvaluating Generalized PCA (Theoretical)...")
            gpca_results = self._evaluate_generalized_pca(X_train, X_test, y_train, y_test)
            dataset_results['methods']['generalized_pca'] = gpca_results
            
            # 4. Create summary
            summary = self._create_dataset_summary(dataset_results)
            dataset_results['summary'] = summary
            
            # 5. Generate visualizations
            self._create_dataset_visualizations(dataset_name, dataset_results)
            
            # Store results
            self.all_results[dataset_name] = dataset_results
            
            # Add to summary
            self.summary_results.append({
                'Dataset': dataset_name,
                'N_Samples': X_train.shape[0] + X_test.shape[0],
                'N_Features': X_train.shape[1],
                'LR_MSE': lr_results.get('test_mse', np.nan),
                'PCA_Best_MSE': pca_results.get('best_test_mse', np.nan),
                'PCA_Best_Components': pca_results.get('best_n_components', np.nan),
                'GPCA_Best_MSE': gpca_results.get('best_test_mse', np.nan),
                'GPCA_Best_Components': gpca_results.get('best_n_components', np.nan),
                'Best_Method': summary.get('best_method', 'Unknown'),
                'Best_MSE': summary.get('best_mse', np.nan)
            })
            
            print(f"\\n✅ Successfully evaluated {dataset_name}")
            print(f"Best method: {summary.get('best_method', 'Unknown')} (MSE: {summary.get('best_mse', np.nan):.6f})")
            
            return dataset_results
            
        except Exception as e:
            print(f"\\n❌ Failed to evaluate {dataset_name}: {str(e)}")
            self.failed_experiments.append({
                'dataset': dataset_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def _evaluate_linear_regression(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate Linear Regression."""
        try:
            k = X_train.shape[1] + 1
            lr = LinearRegressionOperator(k, fit_intercept=True)
            lr.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)
            
            # Metrics
            train_mse = DataMetrics.mse(y_train, y_pred_train)
            test_mse = DataMetrics.mse(y_test, y_pred_test)
            train_r2 = DataMetrics.r2_score(y_train, y_pred_train)
            test_r2 = DataMetrics.r2_score(y_test, y_pred_test)
            
            return {
                'model': lr,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'n_parameters': X_train.shape[1] + 1,  # +1 for intercept
                'operator_info': lr.get_operator_info()
            }
            
        except Exception as e:
            print(f"Linear Regression failed: {e}")
            return {'error': str(e), 'test_mse': np.inf}
    
    def _evaluate_pca_regression(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate PCA Regression with different numbers of components."""
        k = X_train.shape[1] + 1
        max_components = min(X_train.shape[0] - 1, X_train.shape[1])
        
        # Test different numbers of components
        component_range = list(range(1, min(max_components + 1, 11)))  # Test up to 10 components
        
        results = {}
        best_mse = np.inf
        best_config = None
        
        for n_comp in component_range:
            try:
                pca_reg = PCARegressionOperator(k, n_components=n_comp, center=False)  # Already centered
                pca_reg.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = pca_reg.predict(X_train)
                y_pred_test = pca_reg.predict(X_test)
                
                # Metrics
                train_mse = DataMetrics.mse(y_train, y_pred_train)
                test_mse = DataMetrics.mse(y_test, y_pred_test)
                train_r2 = DataMetrics.r2_score(y_train, y_pred_train)
                test_r2 = DataMetrics.r2_score(y_test, y_pred_test)
                
                config_results = {
                    'model': pca_reg,
                    'n_components': n_comp,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'explained_variance_ratio': pca_reg.explained_variance_ratio_.sum() if hasattr(pca_reg, 'explained_variance_ratio_') else np.nan
                }
                
                results[f'n_components_{n_comp}'] = config_results
                
                # Track best
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_config = n_comp
                    
            except Exception as e:
                print(f"PCA with {n_comp} components failed: {e}")
                results[f'n_components_{n_comp}'] = {'error': str(e), 'test_mse': np.inf}
        
        # Summary
        results['best_test_mse'] = best_mse
        results['best_n_components'] = best_config
        results['all_components_tested'] = component_range
        
        return results
    
    def _evaluate_generalized_pca(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate Generalized PCA (Theoretical Implementation)."""
        k = X_train.shape[1] + 1
        max_components = min(X_train.shape[0] - 1, X_train.shape[1])
        
        # Test different numbers of components (fewer due to computational cost)
        component_range = list(range(1, min(max_components + 1, 6)))  # Test up to 5 components
        
        results = {}
        best_mse = np.inf
        best_config = None
        
        for n_comp in component_range:
            try:
                # Use the cleaned theoretical implementation
                gpca = GeneralizedPCARegressionOperator(k, n_components=n_comp, center=False)  # Already centered
                gpca.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = gpca.predict(X_train)
                y_pred_test = gpca.predict(X_test)
                
                # Metrics
                train_mse = DataMetrics.mse(y_train, y_pred_train)
                test_mse = DataMetrics.mse(y_test, y_pred_test)
                train_r2 = DataMetrics.r2_score(y_train, y_pred_train)
                test_r2 = DataMetrics.r2_score(y_test, y_pred_test)
                
                # Additional theoretical metrics
                operator_info = gpca.get_operator_info()
                
                config_results = {
                    'model': gpca,
                    'n_components': n_comp,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'A_sigma': operator_info.get('A_sigma', np.nan),
                    'A_beta_norm': operator_info.get('A_beta_norm', np.nan),
                    'A_mu_frobenius_norm': operator_info.get('A_mu_frobenius_norm', np.nan),
                    'explained_variance_ratio': gpca.explained_variance_ratio_.sum() if hasattr(gpca, 'explained_variance_ratio_') else np.nan
                }
                
                results[f'n_components_{n_comp}'] = config_results
                
                # Track best
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_config = n_comp
                    
            except Exception as e:
                print(f"Generalized PCA with {n_comp} components failed: {e}")
                results[f'n_components_{n_comp}'] = {'error': str(e), 'test_mse': np.inf}
        
        # Summary
        results['best_test_mse'] = best_mse
        results['best_n_components'] = best_config
        results['all_components_tested'] = component_range
        
        return results
    
    def _create_dataset_summary(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for a single dataset."""
        methods = dataset_results['methods']
        
        # Extract best MSE for each method
        lr_mse = methods.get('linear_regression', {}).get('test_mse', np.inf)
        pca_mse = methods.get('pca_regression', {}).get('best_test_mse', np.inf)
        gpca_mse = methods.get('generalized_pca', {}).get('best_test_mse', np.inf)
        
        # Find best method
        mse_comparison = {
            'Linear Regression': lr_mse,
            'PCA Regression': pca_mse,
            'Generalized PCA': gpca_mse
        }
        
        # Remove infinite values for comparison
        valid_mses = {k: v for k, v in mse_comparison.items() if np.isfinite(v)}
        
        if valid_mses:
            best_method = min(valid_mses.items(), key=lambda x: x[1])
            best_method_name, best_mse = best_method
        else:
            best_method_name, best_mse = 'None', np.inf
        
        # Performance improvements over Linear Regression
        improvements = {}
        if np.isfinite(lr_mse):
            for method, mse in mse_comparison.items():
                if method != 'Linear Regression' and np.isfinite(mse):
                    improvement = (lr_mse - mse) / lr_mse * 100
                    improvements[f'{method}_improvement'] = improvement
        
        return {
            'mse_comparison': mse_comparison,
            'best_method': best_method_name,
            'best_mse': best_mse,
            'improvements_over_lr': improvements,
            'valid_methods': list(valid_mses.keys())
        }
    
    def _create_dataset_visualizations(self, dataset_name: str, results: Dict[str, Any]) -> None:
        """Create visualizations for a single dataset."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Evaluation Results: {dataset_name.replace("_", " ").title()}', fontsize=16)
            
            # 1. MSE Comparison
            ax1 = axes[0, 0]
            mse_data = results['summary']['mse_comparison']
            methods = []
            mses = []
            
            for method, mse in mse_data.items():
                if np.isfinite(mse):
                    methods.append(method.replace('_', '\\n'))
                    mses.append(mse)
            
            if mses:
                colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(methods)]
                bars = ax1.bar(methods, mses, color=colors)
                ax1.set_ylabel('Test MSE')
                ax1.set_title('MSE Comparison')
                ax1.tick_params(axis='x', rotation=0)
                
                # Add value labels
                for bar, mse in zip(bars, mses):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                            f'{mse:.4f}', ha='center', va='bottom', fontsize=10)
            else:
                ax1.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=ax1.transAxes)
            
            # 2. PCA Components vs MSE
            ax2 = axes[0, 1]
            pca_results = results['methods'].get('pca_regression', {})
            pca_components = []
            pca_mses = []
            
            for key, data in pca_results.items():
                if key.startswith('n_components_') and 'test_mse' in data:
                    n_comp = int(key.split('_')[-1])
                    test_mse = data['test_mse']
                    if np.isfinite(test_mse):
                        pca_components.append(n_comp)
                        pca_mses.append(test_mse)
            
            if pca_components:
                ax2.plot(pca_components, pca_mses, 'o-', color='lightcoral', linewidth=2, markersize=6)
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Test MSE')
                ax2.set_title('PCA: Components vs MSE')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No PCA results', ha='center', va='center', transform=ax2.transAxes)
            
            # 3. Generalized PCA Components vs MSE
            ax3 = axes[1, 0]
            gpca_results = results['methods'].get('generalized_pca', {})
            gpca_components = []
            gpca_mses = []
            
            for key, data in gpca_results.items():
                if key.startswith('n_components_') and 'test_mse' in data:
                    n_comp = int(key.split('_')[-1])
                    test_mse = data['test_mse']
                    if np.isfinite(test_mse):
                        gpca_components.append(n_comp)
                        gpca_mses.append(test_mse)
            
            if gpca_components:
                ax3.plot(gpca_components, gpca_mses, 's-', color='lightgreen', linewidth=2, markersize=6)
                ax3.set_xlabel('Number of Components')
                ax3.set_ylabel('Test MSE')
                ax3.set_title('Generalized PCA: Components vs MSE')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No Generalized PCA results', ha='center', va='center', transform=ax3.transAxes)
            
            # 4. Method Comparison with Best Components
            ax4 = axes[1, 1]
            comparison_data = []
            comparison_labels = []
            
            # Linear Regression
            lr_mse = results['methods'].get('linear_regression', {}).get('test_mse', np.nan)
            if np.isfinite(lr_mse):
                comparison_data.append(lr_mse)
                comparison_labels.append('Linear\\nRegression')
            
            # Best PCA
            pca_best_mse = pca_results.get('best_test_mse', np.nan)
            pca_best_comp = pca_results.get('best_n_components', 'N/A')
            if np.isfinite(pca_best_mse):
                comparison_data.append(pca_best_mse)
                comparison_labels.append(f'PCA\\n(r={pca_best_comp})')
            
            # Best Generalized PCA
            gpca_best_mse = gpca_results.get('best_test_mse', np.nan)
            gpca_best_comp = gpca_results.get('best_n_components', 'N/A')
            if np.isfinite(gpca_best_mse):
                comparison_data.append(gpca_best_mse)
                comparison_labels.append(f'Gen. PCA\\n(r={gpca_best_comp})')
            
            if comparison_data:
                colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(comparison_data)]
                bars = ax4.bar(comparison_labels, comparison_data, color=colors)
                ax4.set_ylabel('Test MSE')
                ax4.set_title('Best Configuration Comparison')
                
                # Add value labels
                for bar, mse in zip(bars, comparison_data):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                            f'{mse:.4f}', ha='center', va='bottom', fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / f"{dataset_name}_evaluation.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Visualization saved: {plot_path}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to create visualization for {dataset_name}: {e}")
    
    def run_all_evaluations(self) -> None:
        """Run evaluations on all available datasets."""
        print(f"\\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION OF ALL DATASETS")
        print(f"Found {len(self.datasets)} datasets to evaluate")
        print(f"{'='*80}")
        
        successful_evaluations = 0
        
        for dataset_name in sorted(self.datasets.keys()):
            result = self.evaluate_single_dataset(dataset_name)
            if result is not None:
                successful_evaluations += 1
        
        print(f"\\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total datasets: {len(self.datasets)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Failed evaluations: {len(self.failed_experiments)}")
        
        if self.failed_experiments:
            print(f"\\nFailed experiments:")
            for failure in self.failed_experiments:
                print(f"  - {failure['dataset']}: {failure['error']}")
        
        # Create overall summary
        self._create_overall_summary()
        
        # Save all results
        self._save_results()
    
    def _create_overall_summary(self) -> None:
        """Create overall summary across all datasets."""
        if not self.summary_results:
            return
        
        print(f"\\n{'='*80}")
        print(f"OVERALL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        df_summary = pd.DataFrame(self.summary_results)
        
        # Print summary table
        print("\\nDataset Summary:")
        print(df_summary.to_string(index=False, float_format='%.4f'))
        
        # Method performance summary
        print(f"\\n{'='*50}")
        print("METHOD PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        
        # Count best methods
        best_method_counts = df_summary['Best_Method'].value_counts()
        print("\\nBest method frequency:")
        for method, count in best_method_counts.items():
            percentage = count / len(df_summary) * 100
            print(f"  {method}: {count} datasets ({percentage:.1f}%)")
        
        # Average MSE by method
        print("\\nAverage Test MSE by method:")
        lr_avg = df_summary['LR_MSE'].mean() if 'LR_MSE' in df_summary.columns else np.nan
        pca_avg = df_summary['PCA_Best_MSE'].mean() if 'PCA_Best_MSE' in df_summary.columns else np.nan
        gpca_avg = df_summary['GPCA_Best_MSE'].mean() if 'GPCA_Best_MSE' in df_summary.columns else np.nan
        
        print(f"  Linear Regression: {lr_avg:.6f}")
        print(f"  PCA Regression: {pca_avg:.6f}")
        print(f"  Generalized PCA: {gpca_avg:.6f}")
        
        # Create overall visualization
        self._create_overall_visualization(df_summary)
    
    def _create_overall_visualization(self, df_summary: pd.DataFrame) -> None:
        """Create overall summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Overall Evaluation Summary Across All Datasets', fontsize=16)
            
            # 1. Best method frequency
            ax1 = axes[0, 0]
            best_methods = df_summary['Best_Method'].value_counts()
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'yellow', 'purple'][:len(best_methods)]
            wedges, texts, autotexts = ax1.pie(best_methods.values, labels=best_methods.index, 
                                              autopct='%1.1f%%', colors=colors)
            ax1.set_title('Best Method Distribution')
            
            # 2. MSE comparison across datasets
            ax2 = axes[0, 1]
            datasets = df_summary['Dataset'].tolist()
            x_pos = np.arange(len(datasets))
            
            # Only plot methods that have data
            if 'LR_MSE' in df_summary.columns:
                lr_mses = df_summary['LR_MSE'].fillna(np.inf)
                valid_lr = lr_mses != np.inf
                ax2.scatter(x_pos[valid_lr], lr_mses[valid_lr], label='Linear Regression', 
                           color='skyblue', alpha=0.7, s=50)
            
            if 'PCA_Best_MSE' in df_summary.columns:
                pca_mses = df_summary['PCA_Best_MSE'].fillna(np.inf)
                valid_pca = pca_mses != np.inf
                ax2.scatter(x_pos[valid_pca], pca_mses[valid_pca], label='PCA Regression', 
                           color='lightcoral', alpha=0.7, s=50)
            
            if 'GPCA_Best_MSE' in df_summary.columns:
                gpca_mses = df_summary['GPCA_Best_MSE'].fillna(np.inf)
                valid_gpca = gpca_mses != np.inf
                ax2.scatter(x_pos[valid_gpca], gpca_mses[valid_gpca], label='Generalized PCA', 
                           color='lightgreen', alpha=0.7, s=50)
            
            ax2.set_xlabel('Dataset Index')
            ax2.set_ylabel('Test MSE (log scale)')
            ax2.set_yscale('log')
            ax2.set_title('MSE Comparison Across Datasets')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Dataset characteristics
            ax3 = axes[1, 0]
            n_samples = df_summary['N_Samples']
            n_features = df_summary['N_Features']
            scatter = ax3.scatter(n_features, n_samples, c=range(len(df_summary)), 
                                 cmap='viridis', alpha=0.7, s=100)
            ax3.set_xlabel('Number of Features')
            ax3.set_ylabel('Number of Samples')
            ax3.set_title('Dataset Characteristics')
            ax3.grid(True, alpha=0.3)
            
            # Add dataset labels
            for i, dataset in enumerate(datasets):
                ax3.annotate(dataset[:8], (n_features.iloc[i], n_samples.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 4. Average MSE by method
            ax4 = axes[1, 1]
            method_averages = []
            method_names = []
            
            if 'LR_MSE' in df_summary.columns:
                lr_valid = df_summary['LR_MSE'][df_summary['LR_MSE'] != np.inf]
                if len(lr_valid) > 0:
                    method_averages.append(lr_valid.mean())
                    method_names.append('Linear\\nRegression')
            
            if 'PCA_Best_MSE' in df_summary.columns:
                pca_valid = df_summary['PCA_Best_MSE'][df_summary['PCA_Best_MSE'] != np.inf]
                if len(pca_valid) > 0:
                    method_averages.append(pca_valid.mean())
                    method_names.append('PCA\\nRegression')
            
            if 'GPCA_Best_MSE' in df_summary.columns:
                gpca_valid = df_summary['GPCA_Best_MSE'][df_summary['GPCA_Best_MSE'] != np.inf]
                if len(gpca_valid) > 0:
                    method_averages.append(gpca_valid.mean())
                    method_names.append('Generalized\\nPCA')
            
            if method_averages:
                colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(method_averages)]
                bars = ax4.bar(method_names, method_averages, color=colors)
                ax4.set_ylabel('Average Test MSE')
                ax4.set_title('Average MSE Across All Datasets')
                
                # Add value labels
                for bar, avg_mse in zip(bars, method_averages):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                            f'{avg_mse:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save overall plot
            overall_plot_path = self.results_dir / "overall_evaluation_summary.png"
            plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\\n📊 Overall summary visualization saved: {overall_plot_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to create overall visualization: {e}")
    
    def _save_results(self) -> None:
        """Save all results to files."""
        try:
            # Save summary CSV
            if self.summary_results:
                summary_df = pd.DataFrame(self.summary_results)
                summary_path = self.results_dir / "evaluation_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"\\n💾 Summary results saved: {summary_path}")
            
            # Save detailed results as JSON
            results_path = self.results_dir / "detailed_results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for dataset_name, results in self.all_results.items():
                serializable_results[dataset_name] = self._make_json_serializable(results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"💾 Detailed results saved: {results_path}")
            
            # Save failed experiments
            if self.failed_experiments:
                failed_path = self.results_dir / "failed_experiments.json"
                with open(failed_path, 'w') as f:
                    json.dump(self.failed_experiments, f, indent=2)
                print(f"💾 Failed experiments saved: {failed_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items() 
                   if k != 'model'}  # Skip model objects
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)


def main():
    """Main evaluation function."""
    print("🚀 Starting Comprehensive Evaluation of All Methods on All Datasets")
    print("="*80)
    
    # Create evaluation framework
    evaluator = ComprehensiveEvaluation()
    
    # Run all evaluations
    evaluator.run_all_evaluations()
    
    print("\\n🎉 Comprehensive evaluation completed!")
    print(f"📁 Results saved in: {evaluator.results_dir}")
    print("\\nFiles generated:")
    print("  📊 Individual dataset visualizations: {dataset_name}_evaluation.png")
    print("  📊 Overall summary visualization: overall_evaluation_summary.png")
    print("  📄 Summary table: evaluation_summary.csv")
    print("  📄 Detailed results: detailed_results.json")


if __name__ == "__main__":
    main()