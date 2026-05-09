"""
tests/test_outlier_detector.py
-------------------------------
Unit tests for OutlierDetector.
"""

import numpy as np
import pandas as pd
import pytest
from src.outlier_detector import OutlierDetector


@pytest.fixture
def clean_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.normal(0, 1, 200),
        "b": np.random.normal(5, 2, 200),
        "c": np.random.normal(-1, 0.5, 200),
    })


@pytest.fixture
def noisy_df(clean_df):
    df = clean_df.copy()
    df.loc[0, "a"] = 100   # extreme outlier
    df.loc[1, "b"] = -200  # extreme outlier
    return df


class TestOutlierDetectorIQR:

    def test_iqr_detects_outliers(self, noisy_df):
        od = OutlierDetector(method="iqr", treatment="flag")
        result = od.fit_transform(noisy_df)
        assert "is_outlier" in result.columns
        assert result["is_outlier"].sum() >= 1

    def test_iqr_clip_keeps_row_count(self, noisy_df):
        od = OutlierDetector(method="iqr", treatment="clip")
        result = od.fit_transform(noisy_df)
        assert len(result) == len(noisy_df)

    def test_iqr_clip_reduces_extreme_values(self, noisy_df):
        od = OutlierDetector(method="iqr", treatment="clip")
        result = od.fit_transform(noisy_df)
        assert result["a"].max() < 100

    def test_iqr_remove_reduces_row_count(self, noisy_df):
        od = OutlierDetector(method="iqr", treatment="remove")
        result = od.fit_transform(noisy_df)
        assert len(result) < len(noisy_df)


class TestOutlierDetectorZScore:

    def test_zscore_detects_outliers(self, noisy_df):
        od = OutlierDetector(method="zscore", treatment="flag")
        result = od.fit_transform(noisy_df)
        assert result["is_outlier"].sum() >= 1

    def test_zscore_clip_bounds_values(self, noisy_df):
        od = OutlierDetector(method="zscore", treatment="clip", zscore_threshold=3.0)
        result = od.fit_transform(noisy_df)
        assert result["a"].max() < 100


class TestOutlierDetectorIsolationForest:

    def test_if_returns_same_columns(self, noisy_df):
        od = OutlierDetector(method="isolation_forest", treatment="clip", contamination=0.05)
        result = od.fit_transform(noisy_df)
        assert set(noisy_df.columns).issubset(set(result.columns))

    def test_if_clip_keeps_row_count(self, noisy_df):
        od = OutlierDetector(method="isolation_forest", treatment="clip")
        result = od.fit_transform(noisy_df)
        assert len(result) == len(noisy_df)


class TestOutlierDetectorNone:

    def test_none_method_passthrough(self, noisy_df):
        od = OutlierDetector(method="none")
        result = od.fit_transform(noisy_df)
        pd.testing.assert_frame_equal(result, noisy_df)


class TestOutlierSummary:

    def test_summary_returns_dataframe(self, noisy_df):
        od = OutlierDetector(method="iqr")
        od.fit(noisy_df)
        summary = od.outlier_summary(noisy_df)
        assert isinstance(summary, pd.DataFrame)
        assert "outlier_count" in summary.columns
