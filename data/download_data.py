"""
Data Download Script for Heart Disease UCI Dataset.

Downloads the Heart Disease dataset from the UCI ML Repository
and saves it to the data/raw/ directory.
"""

import os
import pandas as pd

def download_heart_disease_data():
    """
    Download the Heart Disease UCI dataset.
    
    Uses the ucimlrepo package to fetch the dataset directly from
    the UCI Machine Learning Repository. Falls back to a direct URL
    if the package fetch fails.
    """
    # Create output directory
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    output_path = os.path.join(raw_dir, "heart_disease.csv")
    
    if os.path.exists(output_path):
        print(f"[INFO] Dataset already exists at: {output_path}")
        return output_path
    
    print("[INFO] Downloading Heart Disease UCI Dataset...")
    
    try:
        # Primary method: ucimlrepo package
        from ucimlrepo import fetch_ucirepo
        
        heart_disease = fetch_ucirepo(id=45)  # Heart Disease dataset ID
        
        # Combine features and targets into a single DataFrame
        df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
        
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"[WARNING] ucimlrepo fetch failed: {e}")
        print("[INFO] Falling back to direct URL download...")
        
        # Fallback: direct download from UCI archive
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        
        column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        
        df = pd.read_csv(url, names=column_names, na_values="?")
        
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    download_heart_disease_data()
