import os
from data_preprocessing import DataPreprocessor
from classical_model import ClassicalModel
from quantum_model import QuantumInspiredModel
from visualizations import ModelVisualizer

def main():
    # Set up directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    visualizations_dir = os.path.join(base_dir, "visualizations")
    
    # Create directories if they don't exist
    for directory in [results_dir, visualizations_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print("Starting disease outbreak dynamics analysis...")
    
    # Step 1: Data Preprocessing
    print("\nStep 1: Preprocessing data...")
    preprocessor = DataPreprocessor(data_dir=data_dir)
    processed_data = preprocessor.process_data()
    print("Data preprocessing completed!")
    
    # Step 2: Classical Model
    print("\nStep 2: Training classical model...")
    classical_model = ClassicalModel(model_type='random_forest')
    X_train, X_test, y_train, y_test = classical_model.prepare_data(
        os.path.join(results_dir, 'preprocessed_dengue_analysis.csv'))
    
    classical_model.train(X_train, y_train)
    predictions, metrics = classical_model.evaluate(X_test, y_test)
    classical_model.save_results(predictions, metrics, results_dir)
    print(f"Classical model completed! RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    # Step 3: Quantum-Inspired Model
    print("\nStep 3: Training quantum-inspired model...")
    quantum_model = QuantumInspiredModel(n_features=X_train.shape[1])
    quantum_model.train(X_train, y_train)
    predictions, metrics = quantum_model.evaluate(X_test, y_test)
    quantum_model.save_results(predictions, metrics, results_dir)
    print(f"Quantum model completed! RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    # Step 4: Generate Visualizations
    print("\nStep 4: Generating visualizations...")
    visualizer = ModelVisualizer(results_dir, visualizations_dir)
    visualizer.generate_all_visualizations()
    print("Visualizations completed!")
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved in: {results_dir}")
    print(f"Visualizations saved in: {visualizations_dir}")

if __name__ == "__main__":
    main()
