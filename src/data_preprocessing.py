"""
Data Preprocessing Module for Heart Disease Dataset.

Handles:
- Loading raw data
- Missing value imputation
- Feature type identification (categorical vs numerical)
- Target variable binarization (0 = no disease, 1 = disease)
- Saving the cleaned dataset
"""

import os
import pandas as pd


def load_raw_data(data_path=None):
    """
    Load the raw heart disease dataset.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the raw CSV file. Defaults to data/raw/heart_disease.csv.
    
    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    if data_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, "data", "raw", "heart_disease.csv")
    
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    return df


def identify_feature_types(df):
    """
    Identify categorical and numerical features in the dataset.
    
    The Heart Disease dataset has specific feature types based on domain knowledge:
    - Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal
    - Numerical: age, trestbps, chol, thalach, oldpeak
    
    Returns
    -------
    tuple of (list, list)
        (categorical_features, numerical_features)
    """
    # Based on UCI dataset documentation
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    
    # Filter to only include columns that exist in the dataframe
    categorical_features = [c for c in categorical_features if c in df.columns]
    numerical_features = [n for n in numerical_features if n in df.columns]
    
    return categorical_features, numerical_features


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Numerical features: impute with median
    - Categorical features: impute with mode
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    
    missing_before = df.isnull().sum()
    total_missing = missing_before.sum()
    
    if total_missing == 0:
        print("[INFO] No missing values found.")
        return df
    
    print(f"[INFO] Found {total_missing} total missing values:")
    for col in missing_before[missing_before > 0].index:
        print(f"       {col}: {missing_before[col]} missing")
    
    categorical_features, numerical_features = identify_feature_types(df)
    
    # Impute numerical features with median
    for col in numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Imputed '{col}' with median: {median_val}")
    
    # Impute categorical features with mode
    for col in categorical_features:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[INFO] Imputed '{col}' with mode: {mode_val}")
    
    # Handle any remaining columns (e.g., target)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"[SUCCESS] All missing values handled. Remaining: {df.isnull().sum().sum()}")
    return df


def binarize_target(df, target_col="num"):
    """
    Binarize the target variable.
    
    The original UCI dataset has target values 0-4:
    - 0: no heart disease
    - 1-4: presence of heart disease (varying severity)
    
    We convert this to binary: 0 (no disease) vs 1 (disease present).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with binarized target.
    """
    df = df.copy()
    
    if target_col not in df.columns:
        print(f"[WARNING] Target column '{target_col}' not found. Checking alternatives...")
        # Common alternative names
        alternatives = ["target", "condition", "heart_disease", "class"]
        for alt in alternatives:
            if alt in df.columns:
                target_col = alt
                print(f"[INFO] Using '{target_col}' as target column.")
                break
    
    if target_col in df.columns:
        unique_vals = sorted(df[target_col].dropna().unique())
        print(f"[INFO] Target column '{target_col}' unique values: {unique_vals}")
        
        if len(unique_vals) > 2:
            # Binarize: 0 stays 0, everything else becomes 1
            df["target"] = (df[target_col] > 0).astype(int)
            if target_col != "target":
                df.drop(columns=[target_col], inplace=True)
            print(f"[INFO] Binarized target: {df['target'].value_counts().to_dict()}")
        else:
            # Already binary or near-binary
            if target_col != "target":
                df.rename(columns={target_col: "target"}, inplace=True)
            # Ensure 0/1 encoding
            df["target"] = df["target"].astype(int)
    else:
        print(f"[ERROR] Could not find target column. Available columns: {list(df.columns)}")
    
    return df


def preprocess_data(data_path=None, save=True):
    """
    Full preprocessing pipeline: load, clean, binarize target.
    
    Parameters
    ----------
    data_path : str, optional
        Path to raw data CSV.
    save : bool
        Whether to save the cleaned dataset.
    
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load
    df = load_raw_data(data_path)
    
    # Standardize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]
    
    print(f"\n[INFO] Dataset columns: {list(df.columns)}")
    print(f"[INFO] Dataset dtypes:\n{df.dtypes}\n")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Binarize target
    df = binarize_target(df)
    
    # Save cleaned data
    if save:
        clean_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(clean_dir, exist_ok=True)
        clean_path = os.path.join(clean_dir, "heart_disease_clean.csv")
        df.to_csv(clean_path, index=False)
        print(f"\n[SUCCESS] Cleaned data saved to: {clean_path}")
    
    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
