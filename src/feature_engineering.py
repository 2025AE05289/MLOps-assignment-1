"""
Feature Engineering Module for Heart Disease Dataset.

Builds a reproducible sklearn Pipeline that handles:
- Numerical features: median imputation + standard scaling
- Categorical features: mode imputation + one-hot encoding

The pipeline is saved as an artifact for full reproducibility.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Feature definitions based on the Heart Disease UCI dataset
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"


def build_preprocessing_pipeline():
    """
    Build a scikit-learn ColumnTransformer preprocessing pipeline.
    
    The pipeline handles:
    - Numerical features: Median imputation → Standard scaling
    - Categorical features: Most-frequent imputation → One-hot encoding
    
    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted-ready preprocessing pipeline.
    """
    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    # Combined transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"  # Drop any columns not specified
    )
    
    return preprocessor


def prepare_features(df, preprocessor=None, fit=True):
    """
    Prepare features for model training/prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw features and target.
    preprocessor : ColumnTransformer, optional
        Pre-fitted preprocessor. If None, a new one is built.
    fit : bool
        If True, fit the preprocessor on the data. If False, only transform.
    
    Returns
    -------
    tuple
        (X_transformed, y, preprocessor)
    """
    # Filter to available features
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [n for n in NUMERICAL_FEATURES if n in df.columns]
    
    print(f"[INFO] Numerical features ({len(available_num)}): {available_num}")
    print(f"[INFO] Categorical features ({len(available_cat)}): {available_cat}")
    
    # Extract target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")
    
    y = df[TARGET_COL].values
    
    # Build or use existing preprocessor
    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline()
    
    if fit:
        X = preprocessor.fit_transform(df)
        print(f"[INFO] Fitted and transformed features. Shape: {X.shape}")
    else:
        X = preprocessor.transform(df)
        print(f"[INFO] Transformed features (using fitted pipeline). Shape: {X.shape}")
    
    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
        print(f"[INFO] Feature names: {list(feature_names)}")
    except Exception:
        feature_names = None
    
    return X, y, preprocessor


def save_preprocessor(preprocessor, output_dir=None):
    """
    Save the fitted preprocessing pipeline to disk.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    output_dir : str, optional
        Directory to save to. Defaults to output/models/.
    """
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "output", "models")
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, path)
    print(f"[SUCCESS] Preprocessing pipeline saved to: {path}")
    return path


def load_preprocessor(path=None):
    """
    Load a saved preprocessing pipeline.
    
    Parameters
    ----------
    path : str, optional
        Path to the saved pipeline. Defaults to output/models/preprocessing_pipeline.joblib.
    
    Returns
    -------
    ColumnTransformer
        Loaded preprocessing pipeline.
    """
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "output", "models", "preprocessing_pipeline.joblib")
    
    preprocessor = joblib.load(path)
    print(f"[INFO] Loaded preprocessing pipeline from: {path}")
    return preprocessor


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import preprocess_data
    
    # Load cleaned data
    df = preprocess_data(save=False)
    
    # Build and fit preprocessing pipeline
    X, y, preprocessor = prepare_features(df)
    
    print(f"\nTransformed feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target distribution: 0={sum(y == 0)}, 1={sum(y == 1)}")
    
    # Save pipeline
    save_preprocessor(preprocessor)
