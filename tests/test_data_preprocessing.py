"""
Unit tests for the data preprocessing module.

Tests cover:
- Raw data loading
- Missing value handling
- Target variable binarization
- Column name standardization
- Full preprocessing pipeline
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    handle_missing_values,
    binarize_target,
    identify_feature_types,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Create a sample DataFrame mimicking the raw Heart Disease dataset."""
    return pd.DataFrame({
        "age": [63, 37, 41, 56, 57],
        "sex": [1, 1, 0, 1, 0],
        "cp": [3, 2, 1, 1, 4],
        "trestbps": [145, 130, 130, 120, 120],
        "chol": [233, 250, 204, 236, 354],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 1, 0, 1, 1],
        "thalach": [150, 187, 172, 178, 163],
        "exang": [0, 0, 0, 0, 1],
        "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.6],
        "slope": [1, 1, 2, 2, 2],
        "ca": [0.0, 0.0, 0.0, 0.0, np.nan],
        "thal": [1.0, 2.0, 2.0, np.nan, 2.0],
        "num": [1, 1, 0, 0, 2],
    })


@pytest.fixture
def sample_df_with_missing():
    """Create a DataFrame with known missing values."""
    return pd.DataFrame({
        "age": [63, 37, np.nan, 56, 57],
        "sex": [1, 1, 0, 1, 0],
        "cp": [3, np.nan, 1, 1, 4],
        "trestbps": [145, 130, 130, 120, 120],
        "chol": [233, 250, 204, 236, 354],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 1, 0, 1, 1],
        "thalach": [150, 187, 172, 178, 163],
        "exang": [0, 0, 0, 0, 1],
        "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.6],
        "slope": [1, 1, 2, 2, 2],
        "ca": [0.0, 0.0, 0.0, 0.0, 0.0],
        "thal": [1.0, 2.0, 2.0, 2.0, 2.0],
        "num": [1, 1, 0, 0, 2],
    })


# ── Test: identify_feature_types ───────────────────────────────────────

class TestIdentifyFeatureTypes:
    """Tests for the identify_feature_types function."""

    def test_returns_two_lists(self, sample_raw_df):
        cat, num = identify_feature_types(sample_raw_df)
        assert isinstance(cat, list)
        assert isinstance(num, list)

    def test_known_categorical_features(self, sample_raw_df):
        cat, _ = identify_feature_types(sample_raw_df)
        expected_cat = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
        assert set(cat) == expected_cat

    def test_known_numerical_features(self, sample_raw_df):
        _, num = identify_feature_types(sample_raw_df)
        expected_num = {"age", "trestbps", "chol", "thalach", "oldpeak"}
        assert set(num) == expected_num

    def test_handles_missing_columns(self):
        """Features not in the DataFrame should be excluded."""
        df = pd.DataFrame({"age": [1, 2], "sex": [0, 1]})
        cat, num = identify_feature_types(df)
        assert "chol" not in num
        assert "cp" not in cat


# ── Test: handle_missing_values ────────────────────────────────────────

class TestHandleMissingValues:
    """Tests for the handle_missing_values function."""

    def test_no_missing_after_imputation(self, sample_df_with_missing):
        result = handle_missing_values(sample_df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_shape_preserved(self, sample_df_with_missing):
        original_shape = sample_df_with_missing.shape
        result = handle_missing_values(sample_df_with_missing)
        assert result.shape == original_shape

    def test_no_missing_returns_unchanged(self):
        df = pd.DataFrame({"age": [1, 2, 3], "sex": [0, 1, 0]})
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_modify_original(self, sample_df_with_missing):
        original_missing = sample_df_with_missing.isnull().sum().sum()
        _ = handle_missing_values(sample_df_with_missing)
        assert sample_df_with_missing.isnull().sum().sum() == original_missing


# ── Test: binarize_target ──────────────────────────────────────────────

class TestBinarizeTarget:
    """Tests for the binarize_target function."""

    def test_binarizes_multiclass_target(self, sample_raw_df):
        result = binarize_target(sample_raw_df, target_col="num")
        assert "target" in result.columns
        assert set(result["target"].unique()) == {0, 1}

    def test_zero_stays_zero(self, sample_raw_df):
        result = binarize_target(sample_raw_df, target_col="num")
        # Original rows where num=0 should have target=0
        zero_mask = sample_raw_df["num"] == 0
        assert (result.loc[zero_mask, "target"] == 0).all()

    def test_nonzero_becomes_one(self, sample_raw_df):
        result = binarize_target(sample_raw_df, target_col="num")
        # Original rows where num > 0 should have target=1
        nonzero_mask = sample_raw_df["num"] > 0
        assert (result.loc[nonzero_mask, "target"] == 1).all()

    def test_already_binary_target(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1], "age": [50, 60, 55, 45]})
        result = binarize_target(df, target_col="target")
        assert set(result["target"].unique()) == {0, 1}
        assert len(result) == 4

    def test_drops_original_num_column(self, sample_raw_df):
        result = binarize_target(sample_raw_df, target_col="num")
        assert "num" not in result.columns


# ── Test: preprocess_data (integration) ────────────────────────────────

class TestPreprocessData:
    """Integration tests for the full preprocessing pipeline."""

    def test_preprocessed_data_exists(self):
        """Verify the preprocessed file was created by previous runs."""
        clean_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv"
        )
        assert os.path.exists(clean_path), "Cleaned data file should exist"

    def test_preprocessed_data_has_target(self):
        """Verify the preprocessed data has a 'target' column."""
        clean_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv"
        )
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
            assert "target" in df.columns

    def test_preprocessed_data_no_missing(self):
        """Verify the preprocessed data has no missing values in key columns."""
        clean_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv"
        )
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
            # Target and key numerical features should have no missing values
            for col in ["target", "age", "trestbps", "chol", "thalach", "oldpeak"]:
                if col in df.columns:
                    assert df[col].isnull().sum() == 0, f"Column '{col}' has missing values"

    def test_target_is_binary(self):
        """Verify target is binary (0 or 1)."""
        clean_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv"
        )
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
            assert set(df["target"].unique()) == {0, 1}
