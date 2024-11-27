import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelVisualizer:
    def __init__(self, results_dir, output_dir):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 12
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest model"""
        # Load feature importances
        importances = pd.read_csv(os.path.join(self.results_dir, 'feature_importances.csv'))
        
        # Create horizontal bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', 
                   y='feature',
                   data=importances,
                   color='skyblue')
        plt.title('Feature Importance in Malaria Incidence Prediction', pad=20)
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        
        # Rotate feature names for better readability
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_vs_actual(self):
        """Plot predicted vs actual values"""
        try:
            # Load the processed data
            data = pd.read_csv(os.path.join(self.results_dir, 'preprocessed_malaria.csv'))
            target_col = 'Incidence of malaria (per 1,000 population at risk)'
            
            # Load predictions
            predictions = pd.read_csv(os.path.join(self.results_dir, 'classical_predictions.csv'))
            
            # Get actual values
            actual = data[target_col].values[:len(predictions)]
            pred_values = predictions['predictions'].values
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(actual, pred_values, alpha=0.5, color='blue', label='Predictions')
            
            # Add perfect prediction line
            min_val = min(min(actual), min(pred_values))
            max_val = max(max(actual), max(pred_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            plt.title('Predicted vs Actual Malaria Incidence', pad=20)
            plt.xlabel('Actual Incidence (per 1,000 population)')
            plt.ylabel('Predicted Incidence (per 1,000 population)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating predictions vs actual plot: {str(e)}")
    
    def plot_geographic_distribution(self):
        """Plot geographic distribution of malaria incidence"""
        try:
            # Load the processed data
            data = pd.read_csv(os.path.join(self.results_dir, 'preprocessed_malaria.csv'))
            target_col = 'Incidence of malaria (per 1,000 population at risk)'
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(data['longitude'], 
                                data['latitude'], 
                                c=data[target_col],
                                cmap='YlOrRd',
                                s=100,
                                alpha=0.6)
            
            plt.colorbar(scatter, label='Malaria Incidence (per 1,000 population)')
            plt.title('Geographic Distribution of Malaria Incidence', pad=20)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'geographic_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating geographic distribution plot: {str(e)}")
    
    def plot_model_metrics(self):
        """Plot model performance metrics"""
        try:
            # Load metrics
            metrics = pd.read_csv(os.path.join(self.results_dir, 'classical_model_metrics.csv'))
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot RMSE
            ax1.bar(['RMSE'], metrics['rmse'], color='skyblue')
            ax1.set_title('Root Mean Square Error')
            ax1.grid(True, alpha=0.3)
            
            # Plot R²
            ax2.bar(['R²'], metrics['r2'], color='lightgreen')
            ax2.set_title('R² Score')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'model_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating model metrics plot: {str(e)}")
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("Generating visualizations...")
        
        self.plot_feature_importance()
        print("- Feature importance plot generated")
        
        self.plot_predictions_vs_actual()
        print("- Predictions vs actual plot generated")
        
        self.plot_geographic_distribution()
        print("- Geographic distribution plot generated")
        
        self.plot_model_metrics()
        print("- Model metrics plot generated")
        
        print(f"\nAll visualizations saved to: {self.output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    visualizations_dir = os.path.join(base_dir, "visualizations")
    
    visualizer = ModelVisualizer(results_dir, visualizations_dir)
    visualizer.generate_all_visualizations()
