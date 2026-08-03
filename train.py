import os
import pandas as pd
from config import Config
from pipeline import DataPreprocessor, CustomerClusteringModel

def run_training():
    config = Config()
    
    # 1. Load raw data
    print(f"Loading data from {config.DATA_PATH}...")
    df_raw = pd.read_csv(config.DATA_PATH)
    df = df_raw.drop(["CustomerID","Gender"], axis=1)
    
    # 2. Preprocess data
    print("Preprocessing features...")
    preprocessor = DataPreprocessor(config)
    X_scaled = preprocessor.fit_transform(df)
    
    preprocessor.save(config.PREPROCESSOR_PATH)
    
    # 3. Train KMeans Model
    print("Training K-Means Model...")
    clustering_model = CustomerClusteringModel(config)
    clustering_model.train(X_scaled)
    
    clustering_model.save(config.MODEL_PATH)
    
    print("Training complete! Model artifacts saved directly in the root directory.")

if __name__ == "__main__":
    run_training()