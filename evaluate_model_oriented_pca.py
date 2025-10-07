"""
Comprehensive Evaluation Script for Model-Oriented Generalized PCA

This script evaluates the performance of the Model-Oriented Generalized PCA
implementation across all available datasets and compares it with baseline methods.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import warnings
from typing import Dict, List, Tuple, Any
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('src')

from src.model_oriented_generalized_pca import ModelOrientedGeneralizedPCA
from src.generalized_pca import GeneralizedPCARegressionOperator  # Previous implementation for comparison
from src.pca import PCARegressionOperator
from src.linear_regression import LinearRegressionOperator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')


class ModelOrientedPCAEvaluator:
    """
    Comprehensive evaluator for Model-Oriented Generalized PCA.
    """
    
    def __init__(self, data_dir: str = "data/processed", results_dir: str = "results/model_oriented"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.datasets = []
        self.results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Find available datasets
        self._discover_datasets()
    
    def _discover_datasets(self):
        """Discover available datasets in the processed data directory."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        for item in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, item)
            if os.path.isdir(dataset_path) and item != '__pycache__':
                # Check if required files exist
                required_files = ['X_train_scaled.csv', 'y_train.csv', 'X_test_scaled.csv', 'y_test.csv']
                if all(os.path.exists(os.path.join(dataset_path, f)) for f in required_files):
                    self.datasets.append(item)
        
        logger.info(f"Found {len(self.datasets)} datasets: {', '.join(self.datasets)}")
    
    def _load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load a dataset and its metadata."""
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        # Load data
        X_train = pd.read_csv(os.path.join(dataset_path, 'X_train_scaled.csv')).values
        y_train = pd.read_csv(os.path.join(dataset_path, 'y_train.csv')).values.ravel()
        X_test = pd.read_csv(os.path.join(dataset_path, 'X_test_scaled.csv')).values
        y_test = pd.read_csv(os.path.join(dataset_path, 'y_test.csv')).values.ravel()
        
        # Load metadata if available
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Loaded {dataset_name}: X_train {X_train.shape}, y_train {y_train.shape}")
        
        return X_train, y_train, X_test, y_test, metadata
    
    def _evaluate_model(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model and return metrics."""
        start_time = time.time()
        
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            fit_time = time.time() - start_time
            
            # Model-specific diagnostics
            diagnostics = {}
            if hasattr(model, 'get_diagnostics'):
                try:
                    diagnostics = model.get_diagnostics()
                except:
                    pass
            
            # Component information
            components_info = {}
            if hasattr(model, 'get_components'):
                try:
                    components = model.get_components()
                    if 'V_H' in components:
                        components_info['n_components_H'] = components['V_H'].shape[1]
                    if 'V_beta' in components:
                        components_info['n_components_beta'] = components['V_beta'].shape[1]
                except:
                    pass
            
            result = {
                'model_name': model_name,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'fit_time': fit_time,
                'success': True,
                'diagnostics': diagnostics,
                'components_info': components_info
            }
            
            logger.info(f"{model_name} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            result = {
                'model_name': model_name,
                'success': False,
                'error': str(e),
                'fit_time': time.time() - start_time
            }
        
        return result
    
    def _create_models(self, n_features: int, n_samples: int) -> Dict[str, Any]:
        """Create model instances for comparison."""
        models = {}
        
        # Model-Oriented Generalized PCA (our new implementation)
        max_r = min(n_features, n_samples - 1, 8)
        r_beta_candidates = list(range(0, min(6, max_r + 1)))
        r_H_candidates = list(range(1, max_r + 1))
        
        models['ModelOrientedPCA'] = ModelOrientedGeneralizedPCA(
            r_beta_candidates=r_beta_candidates,
            r_H_candidates=r_H_candidates,
            max_iter=50,
            tol=1e-6,
            model_selection='bic',
            random_state=42
        )
        
        # Previous Generalized PCA for comparison (using the operator approach)
        try:
            models['GeneralizedPCA'] = GeneralizedPCARegressionOperator()
        except:
            logger.warning("Could not create GeneralizedPCARegressionOperator model")
        
        # Standard PCA + Linear Regression
        try:
            models['PCA+LinearReg'] = PCARegressionOperator(
                n_components=min(min(n_features, n_samples - 1), 10)
            )
        except:
            logger.warning("Could not create PCARegressionOperator model")
        
        # Pure Linear Regression
        try:
            models['LinearRegression'] = LinearRegressionOperator()
        except:
            logger.warning("Could not create LinearRegressionOperator model")
        
        # PLS Regression - wrap sklearn model
        class PLSWrapper:
            def __init__(self, n_components):
                self.model = PLSRegression(n_components=n_components, scale=False)
                
            def fit(self, X, y):
                self.model.fit(X, y)
                return self
                
            def predict(self, X):
                return self.model.predict(X).ravel()
        
        models['PLSRegression'] = PLSWrapper(
            n_components=min(min(n_features, n_samples - 1), 10)
        )
        
        return models
    
    def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Evaluate all models on a single dataset."""
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        # Load dataset
        X_train, y_train, X_test, y_test, metadata = self._load_dataset(dataset_name)
        
        # Create models
        models = self._create_models(X_train.shape[1], X_train.shape[0])
        
        # Evaluate each model
        dataset_results = {
            'dataset_name': dataset_name,
            'metadata': metadata,
            'data_info': {
                'n_train_samples': X_train.shape[0],
                'n_test_samples': X_test.shape[0],
                'n_features': X_train.shape[1],
                'train_target_mean': float(np.mean(y_train)),
                'train_target_std': float(np.std(y_train)),
                'test_target_mean': float(np.mean(y_test)),
                'test_target_std': float(np.std(y_test))
            },
            'model_results': {}
        }
        
        for model_name, model in models.items():
            result = self._evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
            dataset_results['model_results'][model_name] = result
        
        return dataset_results
    
    def run_comprehensive_evaluation(self):
        """Run evaluation on all datasets."""
        logger.info("Starting comprehensive evaluation...")
        
        all_results = []
        
        for dataset_name in self.datasets:
            try:
                dataset_result = self.evaluate_dataset(dataset_name)
                all_results.append(dataset_result)
                self.results[dataset_name] = dataset_result
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                continue
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'detailed_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Saved detailed results to {results_file}")
        
        # Create summary
        self._create_summary()
        
        # Create visualizations
        self._create_visualizations()
        
        return all_results
    
    def _create_summary(self):
        """Create a summary of results across all datasets."""
        summary_data = []
        
        for dataset_name, dataset_result in self.results.items():
            for model_name, model_result in dataset_result['model_results'].items():
                if model_result.get('success', False):
                    row = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'test_r2': model_result['test_r2'],
                        'test_mse': model_result['test_mse'],
                        'test_mae': model_result['test_mae'],
                        'train_r2': model_result['train_r2'],
                        'fit_time': model_result['fit_time'],
                        'n_features': dataset_result['data_info']['n_features'],
                        'n_train_samples': dataset_result['data_info']['n_train_samples']
                    }
                    
                    # Add model-specific metrics
                    if 'diagnostics' in model_result:
                        diag = model_result['diagnostics']
                        if 'kl_divergence' in diag:
                            row['kl_divergence'] = diag['kl_divergence']
                        if 'r_beta_optimal' in diag:
                            row['r_beta_optimal'] = diag['r_beta_optimal']
                        if 'r_H_optimal' in diag:
                            row['r_H_optimal'] = diag['r_H_optimal']
                    
                    summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = os.path.join(self.results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Create performance comparison
        if len(summary_df) > 0:
            perf_summary = summary_df.groupby('model').agg({
                'test_r2': ['mean', 'std', 'count'],
                'test_mse': ['mean', 'std'],
                'fit_time': ['mean', 'std']
            }).round(4)
            
            perf_file = os.path.join(self.results_dir, 'performance_summary.csv')
            perf_summary.to_csv(perf_file)
            
            logger.info("Performance Summary:")
            print(perf_summary)
        
        return summary_df
    
    def _create_visualizations(self):
        """Create comprehensive visualizations."""
        if not self.results:
            return
        
        # Collect data for visualization
        plot_data = []
        for dataset_name, dataset_result in self.results.items():
            for model_name, model_result in dataset_result['model_results'].items():
                if model_result.get('success', False):
                    plot_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'test_r2': model_result['test_r2'],
                        'test_mse': model_result['test_mse'],
                        'fit_time': model_result['fit_time']
                    })
        
        if not plot_data:
            logger.warning("No successful results to plot")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model-Oriented Generalized PCA Evaluation Results', fontsize=16, fontweight='bold')
        
        # R² comparison (box plot equivalent)
        models = df['model'].unique()
        r2_data = [df[df['model'] == model]['test_r2'].values for model in models]
        bp1 = axes[0, 0].boxplot(r2_data, labels=models, patch_artist=True)
        axes[0, 0].set_title('Test R² Distribution by Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Test R²')
        
        # MSE comparison  
        mse_data = [df[df['model'] == model]['test_mse'].values for model in models]
        bp2 = axes[0, 1].boxplot(mse_data, labels=models, patch_artist=True)
        axes[0, 1].set_title('Test MSE Distribution by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('Test MSE')
        axes[0, 1].set_yscale('log')
        
        # Performance by dataset (heatmap equivalent)
        if len(df['dataset'].unique()) > 1:
            pivot_r2 = df.pivot(index='dataset', columns='model', values='test_r2')
            im = axes[1, 0].imshow(pivot_r2.values, cmap='viridis', aspect='auto')
            axes[1, 0].set_xticks(range(len(pivot_r2.columns)))
            axes[1, 0].set_xticklabels(pivot_r2.columns, rotation=45)
            axes[1, 0].set_yticks(range(len(pivot_r2.index)))
            axes[1, 0].set_yticklabels(pivot_r2.index)
            axes[1, 0].set_title('Test R² by Dataset and Model')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 0])
            
            # Add text annotations
            for i in range(len(pivot_r2.index)):
                for j in range(len(pivot_r2.columns)):
                    if not pd.isna(pivot_r2.iloc[i, j]):
                        axes[1, 0].text(j, i, f'{pivot_r2.iloc[i, j]:.3f}', 
                                       ha='center', va='center', color='white')
        
        # Fit time comparison
        time_data = [df[df['model'] == model]['fit_time'].values for model in models]
        bp3 = axes[1, 1].boxplot(time_data, labels=models, patch_artist=True)
        axes[1, 1].set_title('Fit Time Distribution by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylabel('Fit Time (seconds)')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, 'evaluation_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed performance plot for each dataset
        self._create_detailed_plots(df)
        
        logger.info(f"Saved visualizations to {self.results_dir}")
    
    def _create_detailed_plots(self, df: pd.DataFrame):
        """Create detailed plots for each dataset."""
        datasets = df['dataset'].unique()
        
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            
            if len(dataset_df) < 2:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Model Performance on {dataset}', fontsize=14, fontweight='bold')
            
            # R² comparison (bar plot)
            models = dataset_df['model'].values
            r2_values = dataset_df['test_r2'].values
            bars1 = axes[0].bar(range(len(models)), r2_values)
            axes[0].set_title('Test R²')
            axes[0].set_xticks(range(len(models)))
            axes[0].set_xticklabels(models, rotation=45)
            axes[0].set_ylim(0, 1)
            
            # Add values on bars
            for i, v in enumerate(r2_values):
                axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # MSE comparison
            mse_values = dataset_df['test_mse'].values
            bars2 = axes[1].bar(range(len(models)), mse_values)
            axes[1].set_title('Test MSE')
            axes[1].set_xticks(range(len(models)))
            axes[1].set_xticklabels(models, rotation=45)
            
            # Fit time comparison  
            time_values = dataset_df['fit_time'].values
            bars3 = axes[2].bar(range(len(models)), time_values)
            axes[2].set_title('Fit Time (seconds)')
            axes[2].set_xticks(range(len(models)))
            axes[2].set_xticklabels(models, rotation=45)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.results_dir, f'{dataset}_detailed_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_report(self):
        """Create a comprehensive evaluation report."""
        report_file = os.path.join(self.results_dir, 'EVALUATION_REPORT.md')
        
        with open(report_file, 'w') as f:
            f.write("# Model-Oriented Generalized PCA Evaluation Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report presents the evaluation results of the Model-Oriented Generalized PCA ")
            f.write("implementation across multiple datasets, comparing it against baseline methods.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("The Model-Oriented Generalized PCA uses:\n")
            f.write("- Alternating subspace optimization\n")
            f.write("- KL divergence minimization between empirical and model-induced distributions\n")
            f.write("- Joint optimization over regression subspace (V_β) and PCA subspace (V_H)\n")
            f.write("- Model selection via BIC criterion\n\n")
            
            f.write("## Datasets Evaluated\n\n")
            for dataset in self.datasets:
                f.write(f"- {dataset}\n")
            f.write("\n")
            
            f.write("## Models Compared\n\n")
            f.write("1. **ModelOrientedPCA**: Our new implementation\n")
            f.write("2. **GeneralizedPCA**: Previous generalized PCA implementation\n")
            f.write("3. **PCA+LinearReg**: Standard PCA followed by linear regression\n")
            f.write("4. **LinearRegression**: Pure linear regression baseline\n")
            f.write("5. **PLSRegression**: Partial Least Squares regression\n\n")
            
            # Add performance summary if available
            summary_file = os.path.join(self.results_dir, 'evaluation_summary.csv')
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
                
                f.write("## Performance Summary\n\n")
                
                # Overall performance by model
                if len(summary_df) > 0:
                    perf_summary = summary_df.groupby('model').agg({
                        'test_r2': ['mean', 'std', 'count'],
                        'test_mse': ['mean', 'std'],
                        'fit_time': ['mean', 'std']
                    }).round(4)
                    
                    f.write("### Average Performance Across All Datasets\n\n")
                    f.write("| Model | Avg Test R² | Std Test R² | Avg Test MSE | Avg Fit Time (s) |\n")
                    f.write("|-------|-------------|-------------|--------------|------------------|\n")
                    
                    for model in perf_summary.index:
                        avg_r2 = perf_summary.loc[model, ('test_r2', 'mean')]
                        std_r2 = perf_summary.loc[model, ('test_r2', 'std')]
                        avg_mse = perf_summary.loc[model, ('test_mse', 'mean')]
                        avg_time = perf_summary.loc[model, ('fit_time', 'mean')]
                        
                        f.write(f"| {model} | {avg_r2:.4f} | {std_r2:.4f} | {avg_mse:.4f} | {avg_time:.4f} |\n")
                    
                    f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("Please refer to the detailed results and visualizations for comprehensive analysis.\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `detailed_results.json`: Complete evaluation results\n")
            f.write("- `evaluation_summary.csv`: Summary statistics\n")
            f.write("- `performance_summary.csv`: Performance comparison by model\n")
            f.write("- `evaluation_comparison.png`: Overall comparison visualizations\n")
            f.write("- `[dataset]_detailed_comparison.png`: Per-dataset detailed plots\n\n")
        
        logger.info(f"Created evaluation report: {report_file}")


def main():
    """Main evaluation function."""
    logger.info("Starting Model-Oriented Generalized PCA Evaluation")
    
    # Create evaluator
    evaluator = ModelOrientedPCAEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Create report
    evaluator.create_report()
    
    logger.info("Evaluation completed successfully!")
    
    return results


if __name__ == "__main__":
    main()