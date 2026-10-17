"""
Unit tests for the feature engineering module.

Tests cover:
- Preprocessing pipeline construction
- Feature transformation (shape, types)
- Pipeline save/load roundtrip
- Correct handling of categorical & numerical features
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import (
    build_preprocessing_pipeline,
    prepare_features,
    save_preprocessor,
    load_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with all required columns."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age": np.random.randint(30, 80, n),
        "sex": np.random.choice([0.0, 1.0], n),
        "cp": np.random.choice([1.0, 2.0, 3.0, 4.0], n),
        "trestbps": np.random.randint(90, 200, n),
        "chol": np.random.randint(120, 570, n),
        "fbs": np.random.choice([0.0, 1.0], n),
        "restecg": np.random.choice([0.0, 1.0, 2.0], n),
        "thalach": np.random.randint(70, 210, n),
        "exang": np.random.choice([0.0, 1.0], n),
        "oldpeak": np.random.uniform(0, 6.2, n).round(1),
        "slope": np.random.choice([1.0, 2.0, 3.0], n),
        "ca": np.random.choice([0.0, 1.0, 2.0, 3.0], n),
        "thal": np.random.choice([3.0, 6.0, 7.0], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def fitted_pipeline(sample_df):
    """Return a fitted preprocessing pipeline and transformed data."""
    X, y, preprocessor = prepare_features(sample_df, fit=True)
    return X, y, preprocessor


# ── Test: build_preprocessing_pipeline ─────────────────────────────────

class TestBuildPipeline:
    """Tests for build_preprocessing_pipeline."""

    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        pipeline = build_preprocessing_pipeline()
        assert isinstance(pipeline, ColumnTransformer)

    def test_has_numerical_and_categorical_transformers(self):
        pipeline = build_preprocessing_pipeline()
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_numerical_transformer_has_scaler(self):
        pipeline = build_preprocessing_pipeline()
        for name, transformer, _ in pipeline.transformers:
            if name == "num":
                step_names = [s[0] for s in transformer.steps]
                assert "scaler" in step_names
                assert "imputer" in step_names

    def test_categorical_transformer_has_encoder(self):
        pipeline = build_preprocessing_pipeline()
        for name, transformer, _ in pipeline.transformers:
            if name == "cat":
                step_names = [s[0] for s in transformer.steps]
                assert "encoder" in step_names
                assert "imputer" in step_names


# ── Test: prepare_features ─────────────────────────────────────────────

class TestPrepareFeatures:
    """Tests for prepare_features."""

    def test_returns_tuple_of_three(self, sample_df):
        result = prepare_features(sample_df)
        assert len(result) == 3

    def test_X_is_numpy_array(self, fitted_pipeline):
        X, y, _ = fitted_pipeline
        assert isinstance(X, np.ndarray)

    def test_y_is_numpy_array(self, fitted_pipeline):
        _, y, _ = fitted_pipeline
        assert isinstance(y, np.ndarray)

    def test_X_has_correct_rows(self, sample_df, fitted_pipeline):
        X, _, _ = fitted_pipeline
        assert X.shape[0] == len(sample_df)

    def test_X_has_more_columns_than_original_numerical(self, fitted_pipeline):
        """One-hot encoding should expand categorical features."""
        X, _, _ = fitted_pipeline
        # Should have more columns than just the 5 numerical features
        assert X.shape[1] > len(NUMERICAL_FEATURES)

    def test_y_values_are_binary(self, fitted_pipeline):
        _, y, _ = fitted_pipeline
        assert set(np.unique(y)).issubset({0, 1})

    def test_no_nan_in_transformed_features(self, fitted_pipeline):
        X, _, _ = fitted_pipeline
        assert not np.isnan(X).any()

    def test_transform_only_mode(self, sample_df):
        """Test that transform-only mode works with a pre-fitted pipeline."""
        _, _, preprocessor = prepare_features(sample_df, fit=True)
        X2, y2, _ = prepare_features(sample_df, preprocessor=preprocessor, fit=False)
        assert X2.shape[0] == len(sample_df)

    def test_raises_without_target(self):
        """Should raise ValueError if target column is missing."""
        df = pd.DataFrame({"age": [50, 60], "sex": [0, 1]})
        with pytest.raises(ValueError, match="Target column"):
            prepare_features(df)


# ── Test: save/load preprocessor roundtrip ─────────────────────────────

class TestSaveLoadPreprocessor:
    """Tests for saving and loading the preprocessing pipeline."""

    def test_save_creates_file(self, fitted_pipeline):
        _, _, preprocessor = fitted_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_preprocessor(preprocessor, output_dir=tmpdir)
            assert os.path.exists(path)

    def test_load_returns_same_type(self, fitted_pipeline):
        from sklearn.compose import ColumnTransformer
        _, _, preprocessor = fitted_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_preprocessor(preprocessor, output_dir=tmpdir)
            loaded = joblib.load(path)
            assert isinstance(loaded, ColumnTransformer)

    def test_roundtrip_produces_same_output(self, sample_df, fitted_pipeline):
        X_original, _, preprocessor = fitted_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessor(preprocessor, output_dir=tmpdir)
            loaded = load_preprocessor(
                os.path.join(tmpdir, "preprocessing_pipeline.joblib")
            )
            X_loaded, _, _ = prepare_features(sample_df, preprocessor=loaded, fit=False)
            np.testing.assert_array_almost_equal(X_original, X_loaded)


# ── Test: feature constants ────────────────────────────────────────────

class TestFeatureConstants:
    """Tests for feature name constants."""

    def test_numerical_features_is_list(self):
        assert isinstance(NUMERICAL_FEATURES, list)
        assert len(NUMERICAL_FEATURES) > 0

    def test_categorical_features_is_list(self):
        assert isinstance(CATEGORICAL_FEATURES, list)
        assert len(CATEGORICAL_FEATURES) > 0

    def test_no_overlap(self):
        overlap = set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)
        assert len(overlap) == 0, f"Features overlap: {overlap}"

    def test_target_col_defined(self):
        assert TARGET_COL == "target"
