import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class ClassicalModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def prepare_data(self, data_path):
        """Load and prepare data for training"""
        df = pd.read_csv(data_path)
        
        # For malaria dataset, target is incidence
        target_col = [col for col in df.columns if 'incidence' in col.lower()][0]
        
        # All other numeric columns are features
        feature_cols = [col for col in df.columns 
                       if col != target_col 
                       and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the classical model"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'rmse': rmse,
            'r2': r2
        }
        
        return predictions, metrics
    
    def save_results(self, predictions, metrics, output_dir):
        """Save model predictions and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        pd.DataFrame({'predictions': predictions}).to_csv(
            os.path.join(output_dir, 'classical_predictions.csv'), index=False)
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, 'classical_model_metrics.csv'), index=False)
        
        # Save model
        joblib.dump(self.model, os.path.join(output_dir, 'classical_model.joblib'))
        
        # Save feature importances if using random forest
        if self.model_type == 'random_forest':
            importances = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.model.feature_importances_
            })
            importances = importances.sort_values('importance', ascending=False)
            importances.to_csv(
                os.path.join(output_dir, 'feature_importances.csv'), 
                index=False
            )

if __name__ == "__main__":
    # Initialize model
    classical_model = ClassicalModel(model_type='random_forest')
    
    # Prepare data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'results', 'preprocessed_malaria.csv')
    X_train, X_test, y_train, y_test = classical_model.prepare_data(data_path)
    
    # Train model
    classical_model.train(X_train, y_train)
    
    # Evaluate and save results
    predictions, metrics = classical_model.evaluate(X_test, y_test)
    classical_model.save_results(predictions, metrics, os.path.join(base_dir, 'results'))
