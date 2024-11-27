import tensorflow as tf
import tensornetwork as tn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

tn.set_default_backend("tensorflow")

class QuantumInspiredModel:
    def __init__(self, n_features, n_qubits=4):
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.scaler = MinMaxScaler()
        self.build_model()
    
    def build_model(self):
        """Build the quantum-inspired tensor network model"""
        # Initialize trainable parameters
        self.weights = tf.Variable(
            tf.random.uniform([self.n_features, self.n_qubits], 0, 2 * np.pi),
            trainable=True
        )
        self.post_process = tf.Variable(
            tf.random.uniform([self.n_qubits, 1], -1, 1),
            trainable=True
        )
    
    def quantum_feature_map(self, x):
        """Apply quantum-inspired feature mapping"""
        # Reshape input for broadcasting
        x = tf.reshape(x, [-1, self.n_features, 1])
        
        # Create quantum-inspired features
        features = tf.cos(x * self.weights)
        features = tf.reduce_mean(features, axis=1)
        
        return features
    
    def forward(self, x):
        """Forward pass through the model"""
        quantum_features = self.quantum_feature_map(x)
        output = tf.matmul(quantum_features, self.post_process)
        return output
    
    @tf.function
    def train_step(self, x, y):
        """Single training step using gradient tape"""
        with tf.GradientTape() as tape:
            predictions = self.forward(x)
            loss = tf.reduce_mean(tf.square(predictions - y))
        
        # Compute gradients
        gradients = tape.gradient(loss, [self.weights, self.post_process])
        
        # Apply gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        optimizer.apply_gradients(zip(gradients, [self.weights, self.post_process]))
        
        return loss
    
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the quantum-inspired model"""
        # Scale features to [0, 2π]
        X_scaled = self.scaler.fit_transform(X_train) * 2 * np.pi
        
        # Convert to tensors
        X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)
        
        # Training loop
        for epoch in range(epochs):
            loss = self.train_step(X_tensor, y_tensor)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        X_scaled = self.scaler.transform(X) * 2 * np.pi
        X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        predictions = self.forward(X_tensor)
        return predictions.numpy()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        mse = tf.reduce_mean(tf.square(predictions - y_test.reshape(-1, 1)))
        rmse = tf.sqrt(mse)
        
        # Calculate R²
        y_mean = np.mean(y_test)
        ss_tot = np.sum((y_test - y_mean) ** 2)
        ss_res = np.sum((y_test - predictions.flatten()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'rmse': rmse.numpy(),
            'r2': r2
        }
        
        return predictions, metrics
    
    def save_results(self, predictions, metrics, output_dir):
        """Save model predictions and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        pd.DataFrame({'predictions': predictions.flatten()}).to_csv(
            os.path.join(output_dir, 'quantum_predictions.csv'), index=False)
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, 'quantum_model_metrics.csv'), index=False)

def load_and_prepare_data(results_dir):
    """Load preprocessed data and prepare for quantum model."""
    datasets = {}
    for filename in os.listdir(results_dir):
        if filename.startswith('preprocessed_'):
            key = filename.replace('preprocessed_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(results_dir, filename))
            
            # Prepare features
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
            target_col = 'cases' if 'cases' in df.columns else 'incidence'
            
            if target_col in feature_cols:
                feature_cols = feature_cols.drop(target_col)
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            datasets[key] = (X, y, feature_cols)
    
    return datasets

def run_quantum_modeling(results_dir):
    """Main function to run quantum modeling pipeline."""
    # Load preprocessed data
    datasets = load_and_prepare_data(results_dir)
    
    for dataset_name, (X, y, feature_cols) in datasets.items():
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        model = QuantumInspiredModel(n_features=X.shape[1])
        model.train(X_train, y_train)
        
        # Evaluate and save results
        predictions, metrics = model.evaluate(X_test, y_test)
        model.save_results(predictions, metrics, os.path.join(results_dir, dataset_name))

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    run_quantum_modeling(RESULTS_DIR)
