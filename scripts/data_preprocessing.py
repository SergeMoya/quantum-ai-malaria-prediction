import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.results_dir = os.path.join(os.path.dirname(data_dir), "results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_data(self):
        """Process malaria dataset"""
        # Load malaria data
        malaria_data = pd.read_csv(os.path.join(self.data_dir, "malaria_data.csv.csv"))
        
        # Drop geometry column as it's not needed for prediction
        if 'geometry' in malaria_data.columns:
            malaria_data = malaria_data.drop('geometry', axis=1)
        
        # Select relevant features
        feature_cols = [
            'Year',
            'Use of insecticide-treated bed nets (% of under-5 population)',
            'Children with fever receiving antimalarial drugs (% of children under age 5 with fever)',
            'Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)',
            'People using safely managed drinking water services (% of population)',
            'People using safely managed sanitation services (% of population)',
            'Rural population (% of total population)',
            'Urban population (% of total population)',
            'latitude',
            'longitude'
        ]
        
        target_col = 'Incidence of malaria (per 1,000 population at risk)'
        
        # Remove rows where target is NaN
        malaria_data = malaria_data.dropna(subset=[target_col])
        logger.info(f"Data shape after removing NaN targets: {malaria_data.shape}")
        
        # Select features that exist in the dataset
        available_features = [col for col in feature_cols if col in malaria_data.columns]
        logger.info(f"Available features: {available_features}")
        
        # Create feature matrix
        X = malaria_data[available_features]
        y = malaria_data[target_col]
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Combine features and target
        processed_data = pd.concat([X_scaled, y], axis=1)
        
        # Save processed data
        output_path = os.path.join(self.results_dir, "preprocessed_malaria.csv")
        processed_data.to_csv(output_path, index=False)
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Features: {list(X_scaled.columns)}")
        logger.info(f"Target: {target_col}")
        logger.info(f"Saved processed data to: {output_path}")
        
        return processed_data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    preprocessor = DataPreprocessor(data_dir)
    preprocessor.process_data()
